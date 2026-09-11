#!/usr/bin/env python3
"""Create the plan through bd CLI. Default: validate and preview only."""
import argparse
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent

def run(args):
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout or str(args))
    return p.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    tasks = plan['tasks']
    keys = [t['key'] for t in tasks]
    assert len(keys) == len(set(keys)), 'Duplicate task keys'
    seen = set()
    for t in tasks:
        assert set(t['depends_on']) <= seen, 'Missing dependency or non-topological DAG'
        assert (ROOT/t['body']).is_file(), t['body']
        seen.add(t['key'])
    if not args.apply:
        for t in tasks:
            print(f"{t['key']} P{t['priority']} {t['title']} <- {','.join(t['depends_on']) or '-'}")
        print(f'Validated: 1 epic, {len(tasks)} tasks. Use --apply from the FreeToken repository root.')
        return
    if not shutil.which('bd'):
        raise RuntimeError('Install Beads (bd) and initialize the project first.')
    if not pathlib.Path('python/freetoken').is_dir() or not pathlib.Path('.beads').exists():
        raise RuntimeError('Run from initialized FreeToken root (python/freetoken and .beads required).')
    run(['bd','create','--help'])
    state_path = ROOT/'.import-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'plan_id':plan['plan_id'], 'repo':str(pathlib.Path.cwd().resolve()), 'ids':{}, 'edges':[], 'pending':None}
    if state['plan_id'] != plan['plan_id'] or state['repo'] != str(pathlib.Path.cwd().resolve()):
        raise RuntimeError('Plan version or repository differs. Read MIGRATION.md; do not delete prior state.')
    if state.get('pending'):
        raise RuntimeError('Interrupted write: inspect pending in .import-state.json and reconcile with bd before retrying. See README.')
    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        temp.replace(state_path)
    def create(key, title, kind, priority, body, parent=None):
        if key in state['ids']:
            run(['bd','show',state['ids'][key],'--json'])
            return state['ids'][key]
        cmd=['bd','create',title,'--type',kind,'--priority',str(priority),'--body-file',str(body),'--json']
        if parent:
            cmd += ['--parent',parent]
        state['pending']={'operation':'create','key':key,'title':title}
        save()
        result=json.loads(run(cmd))
        if isinstance(result,list) and len(result)==1:
            result=result[0]
        issue_id=result.get('id') if isinstance(result,dict) else None
        if not isinstance(issue_id,str) or not issue_id:
            raise RuntimeError('Unexpected bd output. Reconcile pending create manually.')
        state['ids'][key]=issue_id
        state['pending']=None
        save()
        return issue_id
    epic=create('EPIC','FreeToken Qwen3.8 decode optimization: RTX3090 X299 DDR4-2133','epic',1,ROOT/'PLAN.uk.md')
    for t in tasks:
        create(t['key'],t['title'],t['type'],t['priority'],ROOT/t['body'],epic)
    for t in tasks:
        for dep in t['depends_on']:
            edge=[t['key'],dep]
            if edge in state['edges']:
                continue
            state['pending']={'operation':'dependency','edge':edge}
            save()
            run(['bd','dep','add',state['ids'][t['key']],state['ids'][dep]])
            state['edges'].append(edge)
            state['pending']=None
            save()
    print('Import complete. Epic:',epic)
    print('Run bd ready. Do not start work before all dependencies are imported.')

if __name__ == '__main__':
    main()
