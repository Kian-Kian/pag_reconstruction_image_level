from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1]
def load(path,rgb):
    with Image.open(path) as im:
        a=np.array(im.convert('RGB' if rgb else 'L'),dtype=np.float32)/255
    return torch.from_numpy(a.transpose(2,0,1).copy() if rgb else a[None].copy())
class Pairs(Dataset):
    def __init__(self,split,mode,root=None,require_gb=True):
        self.root=Path(root) if root else ROOT/'data_prepared'/split
        self.require_gb=require_gb; self.mode=mode; self.ids=sorted(f.stem for f in (self.root/'alpha/IPF').glob('*.png'))
        if not self.ids: raise ValueError('No samples: '+str(self.root))
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        n=self.ids[i]; f=n+'.png'
        ipf=load(self.root/'alpha/IPF'/f,True); bc=load(self.root/'alpha/BC'/f,False) if self.mode!='single' else torch.zeros_like(ipf[:1])
        gb=load(self.root/'gamma/GB'/f,False) if self.mode=='gb_only' or (self.mode=='triple' and self.require_gb) else torch.zeros_like(bc)
        if self.mode=='bc_only': ipf=bc.repeat(3,1,1)
        if self.mode=='gb_only': ipf=gb.repeat(3,1,1)
        main=torch.cat([ipf,bc]) if self.mode in ['dual','triple'] else ipf
        return main,bc,gb,load(self.root/'gamma/IPF'/f,True),n
