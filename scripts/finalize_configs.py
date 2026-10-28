from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'metadata/experiment_plan.json').read_text())
def best(names):
    scores=[]
    for n in names:
        run=ROOT/'outputs'/n
        if not (run/'TRAINING_COMPLETE').exists():raise RuntimeError('Incomplete '+n)
        rows=list(csv.DictReader((run/'history.csv').open()))
        scores.append((min(float(r['val_loss']) for r in rows),n))
    return min(scores)[1]
choices={}
for key,names in {'base':plan['hpo'][:4],'lr':['capacity_48']+plan['hpo'][4:7],'weight_decay':['capacity_48']+plan['hpo'][7:],'gb_drop':plan['loss_and_drop'][8:]}.items():
    n=best(names);choices[key]={'run':n,'value':json.loads((ROOT/'configs'/(n+'.json')).read_text())[key]}
base=json.loads((ROOT/'configs/final_dual_seed312.json').read_text());base.update({k:v['value'] for k,v in choices.items()});base['epochs']=200
names=[]
for mode in ['single','dual','triple','bc_only','gb_only']:
    for seed in plan['main_seeds'] if mode in ['single','dual','triple'] else [312]:
        name=f'final_{mode}_seed{seed}';cfg={**base,'name':name,'mode':mode,'seed':seed};f=ROOT/'configs'/(name+'.json')
        if f.exists() and json.loads(f.read_text())!=cfg:raise RuntimeError('Refusing to overwrite '+name)
        f.write_text(json.dumps(cfg,indent=2));names.append(name)
(ROOT/'metadata/final_selection.json').write_text(json.dumps({'choices':choices,'runs':names,'note':'Independent sensitivity choices combined; joint final performance must be measured'},indent=2))
print(names)
