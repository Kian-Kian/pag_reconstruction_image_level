import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse,csv,json,random,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ReconNet
from .data import Pairs,ROOT
from .objectives import objective,metrics
from .visualize import snapshot,save_error

def seed_all(n):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)
def forward(net,main,bc,gb,cfg,training):
    mode=cfg['mode']; xg=None
    if mode=='triple' and training:
        mask=(torch.rand(gb.shape[0],1,1,1,device=gb.device)>=cfg['gb_drop']).to(gb.dtype)
        xg=gb*mask
    return net(main,bc if mode in ['dual','triple'] else None,xg,drop_gb=not training)
def save_rgb(x,path):
    from PIL import Image
    Image.fromarray((x.detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')).save(path)
def evaluate(net,dl,cfg,device,out=None):
    net.eval(); rows=[]; losses=[]
    with torch.no_grad():
        for main,bc,gb,y,names in dl:
            main,bc,gb,y=[x.to(device) for x in [main,bc,gb,y]]
            
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            t=time.perf_counter()
            pred=forward(net,main,bc,gb,cfg,False)
            
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            for i,n in enumerate(names):
                loss,_=objective(pred[i:i+1],y[i:i+1],cfg['weights']); losses.append(loss.item())
                _,parts=objective(pred[i:i+1],y[i:i+1],cfg['weights'])
                rows.append({'sample':n,**metrics(pred[i:i+1],y[i:i+1]),**{k:float(v) for k,v in parts.items() if k!='L1'},'time_s':elapsed/len(names)})
                if out:
                    save_rgb(pred[i],out/(n+'_pred.png'));save_rgb(y[i],out/(n+'_ref.png'));save_error(pred[i:i+1],y[i:i+1],out/n)
    return float(np.mean(losses)),{k:float(np.mean([r[k] for r in rows])) for k in ['MAE','PSNR','SSIM','HueSim','CosSim','SSIM_loss','Hue_loss','Cos_loss']},rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--epochs',type=int);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text()); seed_all(cfg['seed'])
    if args.epochs:cfg['epochs']=args.epochs;cfg['name']+='_smoke'
    out=ROOT/'outputs'/cfg['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:raise RuntimeError('Existing run; use --resume or another name')
    device='cuda'; mode=cfg['mode']; branch=mode if mode in ['single','dual','triple'] else 'single'
    net=ReconNet(base=cfg['base'],branch=branch).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and cfg.get('amp_dtype','fp16')=='fp16')
    ds=Pairs('train',mode);va=Pairs('validation',mode)
    dl=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(va,batch_size=1,num_workers=2,pin_memory=True)
    start=1;best=float('inf')
    if args.resume:
        ck=torch.load(out/'last.pt',weights_only=False,map_location=device)
        if ck['config']!=cfg:raise ValueError('Resume config differs')
        net.load_state_dict(ck['net']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start=ck['epoch']+1;best=ck['best_val']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    source_hash={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src').glob('*.py')}
    (out/'provenance.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':sum(x.numel() for x in net.parameters()),'source_sha256':source_hash,'split_sha256':hashlib.sha256((ROOT/'metadata/splits.json').read_bytes()).hexdigest()},indent=2))
    for epoch in range(start,cfg['epochs']+1):
        net.train();tot=[];part_sums={};seen=0;torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for main,bc,gb,y,_ in dl:
            main,bc,gb,y=[x.to(device,non_blocking=True) for x in [main,bc,gb,y]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=cfg['amp'],dtype=torch.bfloat16 if cfg.get('amp_dtype','fp16')=='bf16' else torch.float16):pred=forward(net,main,bc,gb,cfg,True)
            loss,parts=objective(pred,y,cfg['weights'])
            seen+=len(y)
            for k,v in parts.items():part_sums[k]=part_sums.get(k,0)+float(v.detach())*len(y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();tot.append(loss.item())
        val,mm,_=evaluate(net,vl,cfg,device);improved=val<best;best=min(best,val)
        row={'epoch':epoch,'train_loss':float(np.mean(tot)),'val_loss':val,**{'train_'+k:v/seen for k,v in part_sums.items()},**mm,'epoch_s':time.perf_counter()-t,'peak_vram_GB':torch.cuda.max_memory_allocated()/1e9}
        log=out/'history.csv'
        with log.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if epoch==1:w.writeheader()
            w.writerow(row)
        ck={'net':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_val':best,'config':cfg,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        if improved:torch.save(ck,out/'best.pt')
        vis_every=cfg.get('vis_every',10);cam_every=cfg.get('cam_every',50)
        if epoch==1 or epoch==cfg['epochs'] or (vis_every>0 and epoch%vis_every==0):
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            snapshot(net,va,cfg,out,epoch,cam_every>0 and (epoch%cam_every==0 or epoch==cfg['epochs']))
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        print(json.dumps(row),flush=True)
    (out/'TRAINING_COMPLETE').write_text('complete')
if __name__=='__main__':main()
