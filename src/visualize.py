import json,csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw

def rgb(x):
    a=x.detach().float().cpu()
    if a.ndim==4:a=a[0]
    if a.shape[0]==1:a=a.repeat(3,1,1)
    return (a[:3].clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
def color(a):
    a=np.clip(a,0,1)
    return np.stack([np.minimum(1,2*a),1-np.abs(2*a-1),np.maximum(0,1-2*a)],-1)
def save_error(pred,ref,dest):
    err=(pred-ref).abs().mean(1)[0].detach().float().cpu().numpy()
    np.save(str(dest)+'_mae.npy',err)
    Image.fromarray((color(err)*255).astype('uint8')).save(str(dest)+'_error.png')
def comparison(main,bc,ref,pred,path):
    arrays=[rgb(main),rgb(bc),rgb(ref),rgb(pred)]
    w,h=arrays[0].shape[1],arrays[0].shape[0]
    canvas=Image.new('RGB',(w*4,h+24),'white');draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate(zip(arrays,['Input','BC','Reference','Prediction'])):
        canvas.paste(Image.fromarray(a),(i*w,24));draw.text((i*w+5,5),label,fill='black')
    canvas.save(path)
def cam(net,main,bc,gb,ref,mode,dest,enable_gb=False):
    dest.mkdir(parents=True,exist_ok=True);holders={};handles=[]
    layers={'main_bottleneck':net.bott_rest[0],'main_u3':net.u3,'main_u2':net.u2,'main_u1':net.u1}
    if net.aux_bc is not None:
        layers.update({f'bc_l{i+1}':net.aux_bc.enc[j] for i,j in enumerate([1,2,3,4])})
    if enable_gb and net.aux_gb is not None:
        layers.update({'gb_l1':net.aux_gb.stem[1],'gb_l2':net.aux_gb.d1,'gb_l3':net.aux_gb.d2,'gb_l4':net.aux_gb.d3})
    def hook(key):
        def capture(module,inputs,out):
            holders[key]=out;out.retain_grad()
        return capture
    try:
        for key,layer in layers.items():handles.append(layer.register_forward_hook(hook(key)))
        net.eval();net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred=net(main,bc if mode in ['dual','triple'] else None,gb if enable_gb else None,drop_gb=not enable_gb)
            target=pred.mean();target.backward()
        for key,a in holders.items():
            g=a.grad
            if g is None:raise RuntimeError('Missing gradient '+key)
            raw=(g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True).detach().float()
            raw=F.interpolate(raw,size=main.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            np.save(dest/(key+'_raw.npy'),raw)
            pos=np.maximum(raw,0);norm=pos/(pos.max()+1e-12);np.save(dest/(key+'_normalized.npy'),norm)
            heat=(color(norm)*255).astype('uint8');background=rgb(gb if key.startswith('gb') else bc if key.startswith('bc') else main)
            Image.fromarray(heat).save(dest/(key+'_heat.png'))
            Image.fromarray((.5*background+.5*heat).astype('uint8')).save(dest/(key+'_overlay.png'))
        (dest/'metadata.json').write_text(json.dumps({'target':'mean predicted RGB intensity','gb_enabled':enable_gb,'interpretation':'output sensitivity; independently normalized maps cannot compare absolute importance','layers':list(holders)},indent=2))
    finally:
        for h in handles:h.remove()
        net.zero_grad(set_to_none=True)
def snapshot(net,ds,cfg,out,epoch,with_cam):
    dest=out/'visualizations'/f'epoch_{epoch:03d}';dest.mkdir(parents=True,exist_ok=True)
    names=[]
    for i in range(min(cfg.get('vis_samples',3),len(ds))):
        main,bc,gb,ref,n=ds[i];names.append(n)
        main,bc,gb,ref=[x.unsqueeze(0).cuda() for x in [main,bc,gb,ref]]
        with torch.no_grad():pred=net(main,bc if cfg['mode'] in ['dual','triple'] else None,None,drop_gb=True)
        np.savez_compressed(dest/(n+'_arrays.npz'),prediction=pred[0].float().cpu().numpy(),reference=ref[0].cpu().numpy())
        comparison(main,bc,ref,pred,dest/(n+'_comparison.png'));save_error(pred,ref,dest/n)
        if with_cam:
            cam(net,main,bc,gb,ref,cfg['mode'],dest/n/'inference_cam')
            if cfg['mode']=='triple':cam(net,main,bc,gb,ref,cfg['mode'],dest/n/'gb_enabled_cam',True)
    (dest/'samples.json').write_text(json.dumps(names))
