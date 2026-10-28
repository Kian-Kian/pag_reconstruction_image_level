import torch
import torch.nn.functional as F

def ssim(a,b):
    ma=F.avg_pool2d(a,3,1,1); mb=F.avg_pool2d(b,3,1,1)
    va=F.avg_pool2d(a*a,3,1,1)-ma*ma; vb=F.avg_pool2d(b*b,3,1,1)-mb*mb
    cov=F.avg_pool2d(a*b,3,1,1)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()
def hue(x):
    mx,idx=x.max(1); mn=x.min(1).values; delta=mx-mn
    safe=delta.clamp_min(1e-8); r,g,b=x.unbind(1)
    h=torch.where(idx==0,((g-b)/safe).remainder(6),torch.where(idx==1,(b-r)/safe+2,(r-g)/safe+4))/6
    return torch.where(delta>1e-8,h,torch.zeros_like(h))
def terms(a,b):
    a=a.float(); b=b.float()
    dh=(hue(a)-hue(b)).abs(); dh=torch.minimum(dh,1-dh).mean()
    cos=F.cosine_similarity(a,b,dim=1,eps=1e-8).mean()
    return {'L1':F.l1_loss(a,b),'SSIM_loss':1-ssim(a,b),'Hue_loss':dh,'Cos_loss':1-cos}
def objective(a,b,weights):
    t=terms(a,b)
    return t['L1']+sum(weights[k]*t[k+'_loss'] for k in ['SSIM','Hue','Cos']),t
@torch.no_grad()
def metrics(a,b):
    t=terms(a,b)
    return {'MAE':t['L1'].item(),'PSNR':(-10*torch.log10(F.mse_loss(a.float(),b.float()).clamp_min(1e-12))).item(),'SSIM':1-t['SSIM_loss'].item(),'HueSim':1-t['Hue_loss'].item(),'CosSim':1-t['Cos_loss'].item()}
