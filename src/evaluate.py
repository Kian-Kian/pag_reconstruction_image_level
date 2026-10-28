import argparse,csv,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import Pairs,ROOT,load
from .model import ReconNet
from .train import evaluate,save_rgb
from .visualize import cam

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--external',type=Path);ap.add_argument('--predict-only',action='store_true');ap.add_argument('--save-cam',action='store_true');ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');ap.add_argument('--output',type=Path);ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    run=ROOT/'outputs'/args.run;ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config']
    mode=cfg['mode'];branch=mode if mode in ['single','dual','triple'] else 'single'
    net=ReconNet(base=cfg['base'],branch=branch).to(device);net.load_state_dict(ck['net'],strict=True);net.eval()
    if args.predict_only and (args.external is None or mode not in ['single','dual','triple']):
        raise ValueError('--predict-only requires external data and a primary IPF model')
    if args.external and not (args.external/'alpha/IPF').is_dir():
        raise ValueError('Missing external alpha/IPF directory')
    out=args.output if args.output else run/('external_'+args.external.resolve().name if args.external else 'test')
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError('Output directory is not empty; choose --output or explicitly use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    if args.predict_only:
        if args.external is None:raise ValueError('--predict-only requires --external')
        if mode not in ['single','dual','triple']:raise ValueError('External prediction supports primary IPF models')
        with torch.no_grad():
            files=sorted((args.external/'alpha/IPF').glob('*.png'))
            if not files:raise ValueError('No external PNG samples found')
            for f in files:
                ipf=load(f,True).unsqueeze(0).to(device);bc=load(args.external/'alpha/BC'/f.name,False).unsqueeze(0).to(device) if mode!='single' else None
                main=torch.cat([ipf,bc],1) if bc is not None else ipf
                save_rgb(net(main,bc,None,drop_gb=True)[0],out/(f.stem+'_pred.png'))
        return
    ds=Pairs('test',mode,root=args.external,require_gb=False);loss,means,rows=evaluate(net,DataLoader(ds,batch_size=1),cfg,device,out)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'checkpoint_epoch':ck['epoch'],'loss':loss,'n':len(rows),**means},indent=2))
    if args.save_cam:
        for i in range(len(ds)):
            main,bc,gb,y,n=ds[i];main,bc,gb,y=[x.unsqueeze(0).to(device) for x in [main,bc,gb,y]]
            cam(net,main,bc,gb,y,mode,out/'cams'/n)
    print(means)
if __name__=='__main__':main()
