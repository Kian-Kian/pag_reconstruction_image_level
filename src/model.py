import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _zeros_like(b, c, h, w, device, dtype):
    return torch.zeros(b, c, h, w, device=device, dtype=dtype)

class CBAM(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // r), ch, bias=False)
        )
        self.s = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        w   = torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)
        x   = x * w
        y   = torch.sigmoid(self.s(torch.cat([x.mean(1, True), x.max(1, True)[0]], 1)))
        return x * y

class ConvBNReLU(nn.Module):
    def __init__(self, i: int, o: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(o)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvBNReLU(ch, ch)
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + x)

class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for r in rates[1:]
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(self.blocks), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.blocks], 1))

class TransformerBottleneck(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)              
        q = self.n1(t)
        a, _ = self.attn(q, q, q)
        t = t + a
        t = t + self.ff(self.n2(t))
        return t.transpose(1, 2).view(B, C, H, W)

class Down(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c = ConvBNReLU(i, o)
        self.r = ResBlock(o)
        self.cb = CBAM(o)
    def forward(self, x):
        return self.cb(self.r(self.c(self.pool(x))))

class Up(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c = ConvBNReLU(i, o)
        self.r = ResBlock(o)
        self.cb = CBAM(o)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], 1)
        return self.cb(self.r(self.c(x)))

class MainEncoder(nn.Module):
    def __init__(self, base: int = 48, in_ch: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, base),
            ResBlock(base),
            CBAM(base)
        )
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        return s0, s1, s2, s3

class AuxBCEncoder(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBNReLU(1, base),
            ResBlock(base),
            Down(base, base * 2),
            Down(base * 2, base * 4),
            Down(base * 4, base * 8),
            CBAM(base * 8)
        )
    def forward(self, x):
        return self.enc(x)                                                        

class AuxGBEncoder(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, base),
            ResBlock(base),
            CBAM(base)
        )
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
    def forward(self, x):
        g0 = self.stem(x)
        g1 = self.d1(g0)
        g2 = self.d2(g1)
        g3 = self.d3(g2)
        return g0, g1, g2, g3                             

class ReconNet(nn.Module):
    def __init__(self, base: int = 48, out_ch: int = 3, branch: str = "single"):
        super().__init__()
        assert branch in {"single", "dual", "triple"}
        self.branch = branch
        self.base = base

        in_ch_main = 3 if branch == "single" else 4
        self.main = MainEncoder(base, in_ch=in_ch_main)

        if branch in ("dual", "triple"):
            self.aux_bc = AuxBCEncoder(base // 2)                                
        else:
            self.aux_bc = None
        if branch == "triple":
            self.aux_gb = AuxGBEncoder(base // 2)                      
        else:
            self.aux_gb = None

        main_bott_ch = base * 8                               
        aux_each_bott = base * 4                                       
        if branch == "single":
            aspp_in = main_bott_ch                              
        elif branch == "dual":
            aspp_in = main_bott_ch + aux_each_bott                   
        else:          
            aspp_in = main_bott_ch + aux_each_bott * 2                         

        self.aspp = ASPP(aspp_in, base * 8)                                 
        self.bott_rest = nn.Sequential(
            TransformerBottleneck(base * 8),
            ResBlock(base * 8)
        )

        self.u3 = Up(i=base * 8 + base * 4, o=base * 4)
        self.u2 = Up(i=base * 4 + base * 2, o=base * 2)
        self.u1 = Up(i=base * 2 + base,   o=base)
        self.head = nn.Sequential(
            ConvBNReLU(base + base, base),
            nn.Conv2d(base, out_ch, 1)
        )

    @torch.no_grad()
    def _auto_merge_bc_to_main(self, xa: torch.Tensor, xb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        If model branch != 'single' and xa has only 3 channels while xb is provided,
        we merge xb into xa to form 4-channel main input automatically.
        """
        if self.branch != "single" and xa.shape[1] == 3 and xb is not None:
            if xb.shape[1] != 1:
                                                                                 
                xb_1 = xb.mean(dim=1, keepdim=True)
            else:
                xb_1 = xb
                                                    
            if xb_1.shape[2:] != xa.shape[2:]:
                xb_1 = F.interpolate(xb_1, size=xa.shape[2:], mode="bilinear", align_corners=False)
            xa = torch.cat([xa, xb_1], dim=1)
        return xa

    def forward(self, xa: torch.Tensor, xb: Optional[torch.Tensor] = None, xg: Optional[torch.Tensor] = None,
                drop_bc: bool = False, drop_gb: bool = False) -> torch.Tensor:
                                                                                    
        xa = self._auto_merge_bc_to_main(xa, xb)

        s0, s1, s2, s3 = self.main(xa)                             

        B, _, H8, W8 = s3.shape
        device, dtype = s3.device, s3.dtype

        if self.branch in ("dual", "triple") and (not drop_bc) and (xb is not None):
            bc_bott = self.aux_bc(xb)                                                     
            if bc_bott.shape[2:] != (H8, W8):
                bc_bott = F.interpolate(bc_bott, size=(H8, W8), mode="bilinear", align_corners=False)
        elif self.branch in ("dual", "triple"):
            bc_bott = _zeros_like(B, self.base * 4, H8, W8, device, dtype)
        else:
            bc_bott = None

        if self.branch == "triple" and (not drop_gb) and (xg is not None):
            _, _, _, g3 = self.aux_gb(xg)                                                 
            if g3.shape[2:] != (H8, W8):
                g3 = F.interpolate(g3, size=(H8, W8), mode="bilinear", align_corners=False)
        elif self.branch == "triple":
            g3 = _zeros_like(B, self.base * 4, H8, W8, device, dtype)
        else:
            g3 = None

        feats = [s3]
        if self.branch in ("dual", "triple"):
            feats.append(bc_bott)
        if self.branch == "triple":
            feats.append(g3)
        b_in = torch.cat(feats, dim=1)

        b = self.aspp(b_in)
        b = self.bott_rest(b)

        x = self.u3(b, s2)
        x = self.u2(x, s1)
        x = self.u1(x, s0)
        out = self.head(torch.cat([x, s0], 1))
        return torch.sigmoid(out)

