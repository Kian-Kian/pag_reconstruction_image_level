import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('names',nargs='+');args=ap.parse_args()
    for name in args.names:
        cfg=ROOT/'configs'/(name+'.json');run=ROOT/'outputs'/name
        if not (run/'TRAINING_COMPLETE').exists():
            cmd=[sys.executable,'-m','src.train','--config',str(cfg)]
            if (run/'last.pt').exists():cmd+=['--resume']
            subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
