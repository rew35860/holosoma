"""Resume any manifest clip with a shared preset; no sequence-specific weights."""
import argparse,os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('sequence')
parser.add_argument('--output',help='Study folder containing cohort.json and protocol.json')
parser.add_argument('--variant',choices=['v3','recommended','current','tip_relaxed','surface_emphasis'],default='v3',
                    help='v3 reproduces the tuned shared preset; historical variant names remain supported')
args=parser.parse_args()
if args.output:os.environ['V2_CROSS_OUT']=args.output
import worker
r=worker.row(args.sequence)
variant=args.variant
if variant=='v3':variant='surface_emphasis'
if variant=='recommended':
    path=worker.OUT/'recommended_preset.json'
    variant=worker.read(path)['variant'] if path.exists() else 'surface_emphasis'
result=None
for name in ('prepare_inputs','body','prepare'):
    if not worker.stage(args.sequence,name,'current'):
        result=dict(sequence=args.sequence,status='failed',stage=name);break
if result is None:result=worker.process(r,variant)
print(result,flush=True)
raise SystemExit(0 if result['status']=='complete' else 1)
