from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/plot_data';OUT.mkdir(parents=True,exist_ok=True)
def write(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
history=[];tests=[];summaries=[];sources=[]
for run in sorted((ROOT/'outputs').iterdir()):
    if not run.is_dir() or run.name.startswith('archive_') or not (run/'config.json').exists():continue
    cfg=json.loads((run/'config.json').read_text())
    category='validation' if run.name.startswith('validation_') or run.name.endswith('_smoke') else 'experiment'
    tags={'run':run.name,'category':category,'mode':cfg['mode'],'amp_dtype':cfg.get('amp_dtype','fp16'),'seed':cfg['seed'],'base':cfg['base'],'lr':cfg['lr'],'weight_decay':cfg['weight_decay'],'gb_drop':cfg['gb_drop']}
    for f,target in [(run/'history.csv',history),(run/'test/metrics.csv',tests)]:
        if f.exists():
            rows=list(csv.DictReader(f.open()));target.extend([{**tags,**r} for r in rows]);sources.append(str(f.relative_to(ROOT)))
    if (run/'test/summary.json').exists():summaries.append({**tags,**json.loads((run/'test/summary.json').read_text())})
write('all_history.csv',history);write('all_test_samples.csv',tests);write('experiment_summary.csv',summaries)
(OUT/'source_files.json').write_text(json.dumps(sources,indent=2))
print('Exported',len(history),'history rows and',len(tests),'test rows')
