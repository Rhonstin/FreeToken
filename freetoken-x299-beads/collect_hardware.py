#!/usr/bin/env python3
"""Read-only evidence collection, except writing the requested output file."""
import argparse,json,pathlib,subprocess,datetime,os
def capture(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        return {'argv':argv,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'argv':argv,'exit_code':None,'error':str(e)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=pathlib.Path(a.out)
    if out.exists():raise SystemExit('Refusing to overwrite existing hardware evidence')
    commands=[['lscpu','-J'],['lscpu','-e=CPU,CORE,SOCKET,NODE,ONLINE'],['lsblk','-J','-o','NAME,TYPE,SIZE,ROTA,TRAN,MODEL'],['nvidia-smi','--query-gpu=name,memory.total,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current','--format=csv'],['uname','-sr']]
    result={'schema_version':2,'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observations':[capture(c) for c in commands],'ram_channels':None,'ram_channels_reason':'Requires DIMM/controller evidence; not inferred from platform','configured_ram_speed':None,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}
    f=pathlib.Path('/proc/meminfo');result['meminfo']=f.read_text() if f.exists() else None
    result['link_note']='Current link may be idle. Repeat during controlled benchmark into a different evidence file.'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
