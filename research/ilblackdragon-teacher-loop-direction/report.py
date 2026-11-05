"""Export aggregates only; raw public reasoning traces remain local."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def summarize(rows):
    if not rows:
        return None
    return {'n':len(rows), 'correct':sum(r['score'] for r in rows),
            'accuracy':100*statistics.mean(r['score'] for r in rows),
            'loop_count':sum(r['loop_detected'] for r in rows),
            'mean_tail_repetition_coverage':statistics.mean(r['tail_repetition_coverage'] for r in rows),
            'reasoning_closed_count':sum(r['reasoning_closed'] for r in rows),
            'truncated_count':sum(r['truncated'] for r in rows),
            'total_generated_tokens':sum(r['tokens'] for r in rows),
            'generation_seconds':sum(r['seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();w=args.workspace.resolve()
    stages={s:read(w/f'local/{s}.generations.jsonl') for s in ['baseline','teacher','candidate']}
    selected=json.loads((w/'local/selected.json').read_text())
    outcome=json.loads((w/'local/outcome.json').read_text()) if (w/'local/outcome.json').exists() else None
    report={'workspace':json.loads((w/'delta-workspace.json').read_text()),
            'validation_sha256':hashlib.sha256((w/'validation.json').read_bytes()).hexdigest(),
            'scope':'Public local diagnostic estimates; not official League scores',
            'outcome':outcome,
            'selection':[{k:r[k] for k in ['id','dataset_index','manifest_position','doc_sha256','split']} for r in selected],
            'aggregates':{s:{split:summarize([r for r in rows if r['split']==split]) for split in ['development','holdout']} for s,rows in stages.items()},
            'examples':{s:[{k:v for k,v in r.items() if k not in ['text','token_ids','loop_positions']} for r in rows] for s,rows in stages.items()},
            'runtime':{p.stem:json.loads(p.read_text()) for p in sorted((w/'local').glob('*-runtime.json'))},
            'candidate_official_validation_score':None,'candidate_official_test_score':None,
            'submission_id':None,'submission_url':None}
    if (w/'local/edit.json').exists():
        report['edit']=json.loads((w/'local/edit.json').read_text())
    (w/'local/results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['aggregates'],indent=2))


if __name__=='__main__':
    main()
