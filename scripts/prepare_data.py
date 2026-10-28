from pathlib import Path
import json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def transform(im, variant):
    ops={'rot90':Image.Transpose.ROTATE_90,'rot180':Image.Transpose.ROTATE_180,'rot270':Image.Transpose.ROTATE_270,'flip_h':Image.Transpose.FLIP_LEFT_RIGHT,'flip_v':Image.Transpose.FLIP_TOP_BOTTOM}
    if variant=='original': return im.copy()
    if variant in ops: return im.transpose(ops[variant])
    row,col=map(int,variant[-2:]); w,h=im.size
    return im.crop((col*(w//2),row*(h//2),(col+1)*(w//2),(row+1)*(h//2))).resize((w,h),Image.Resampling.BILINEAR)
def main():
    splits=json.loads((ROOT/'metadata/splits.json').read_text())
    for split in ['train','validation','test']:
        for item in splits[split]:
            for modality in ['alpha/IPF','alpha/BC','gamma/IPF','gamma/GB']:
                src=ROOT/'dataset'/modality/(item['source_id']+'.png')
                dst=ROOT/'data_prepared'/split/modality/(item['sample_id']+'.png')
                dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():
                    with Image.open(src) as im: transform(im,item['variant']).save(dst)
        print(split,len(splits[split]),flush=True)
if __name__=='__main__': main()
