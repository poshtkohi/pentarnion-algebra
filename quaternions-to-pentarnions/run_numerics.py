#!/usr/bin/env python3
"""Build and run the Part I/II numerical experiments without changing reference data.

Run --task all for the declared full numerical recipes, or --task smoke for
short numerical tests. All command lines, return codes, software versions and
source hashes are recorded. Long sensitive trajectories are compared, not
assumed bitwise reproducible. --plots with --task all plots the NEW records.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RIC=ROOT/'part-II/code/riccati_dynamics';SPIN=ROOT/'part-I/code/spin_dynamics'
TASKS=['smoke','scans','peaks','spectra','dense','branches','validation','spin','spin-scan','all']
SOURCES={p.stem:p for p in list(RIC.rglob('*.cpp'))+list(SPIN.glob('*.cpp'))}


def digest(path:Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()
def reference_hashes() -> dict:
    return {str(p.relative_to(ROOT)):digest(p) for part in ('I','II') for p in sorted((ROOT/f'part-{part}/data').rglob('*')) if p.is_file()}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS,default='smoke')
    parser.add_argument('--output',type=Path,help='A NEW directory for generated data and logs')
    parser.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--compiler',help='Path/name of a g++ or clang++ C++17 compiler')
    parser.add_argument('--build-profile',choices=['recorded','portable','native'],default='recorded',help='recorded: original native flags for three scan/spectrum sources; portable: omit native flags; native: apply to every driver')
    parser.add_argument('--plots',action='store_true',help='After --task all, plot the regenerated spin and Riccati data')
    parser.add_argument('--timeout',type=float,default=3600,help='Seconds allowed for each program')
    args=parser.parse_args()
    if args.threads<1 or args.timeout<=0:parser.error('Threads and timeout must be positive')
    if args.plots and args.task!='all':parser.error('--plots requires --task all')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(args.output or ROOT/'outputs/numerics'/stamp).expanduser().resolve()
    if out.exists():parser.error('Choose a new --output directory; existing records are never overwritten')
    # Reference trees, source trees and documentation are all read-only inputs.
    if ROOT in out.parents and 'outputs' not in out.relative_to(ROOT).parts:parser.error('Within this repository, use an outputs/ directory')
    compiler=args.compiler or shutil.which('g++') or shutil.which('c++') or shutil.which('clang++')
    if not compiler:parser.error('A C++17 compiler is required (g++ or clang++ on PATH, or --compiler)')
    out.mkdir(parents=True);logs=out/'logs';logs.mkdir();build=out/'build';build.mkdir()
    pi=out/'generated/part-I/data';pii=out/'generated/part-II/data'
    for p in [pi/'spin',pi/'spin_dense',pii/'riccati/periodic_branches',pii/'riccati/validation',pii/'riccati_dense',pii/'time_switching']:p.mkdir(parents=True,exist_ok=True)
    before=reference_hashes();events=[];binaries={};start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def run(name,cmd,extra_env=None):
        cmd=list(map(str,cmd));t=time.monotonic();print('RUN',name,flush=True)
        with (logs/(name+'.log')).open('w',encoding='utf-8') as f:
            f.write('COMMAND: '+' '.join(cmd)+'\n\n');f.flush()
            p=subprocess.run(cmd,cwd=ROOT,env={**env,**(extra_env or {})},stdout=f,stderr=subprocess.STDOUT,timeout=args.timeout)
        events.append({'task':name,'command':cmd,'returncode':p.returncode,'elapsed_seconds':round(time.monotonic()-t,3),'log':name+'.log'})
        if p.returncode:raise RuntimeError(f'{name} returned {p.returncode}; see {logs/(name+".log")}')
        print('PASS',name,flush=True)
    def binary(name):
        if name not in binaries:
            dest=build/(name+('.exe' if os.name=='nt' else ''));cmd=[compiler,'-O3','-std=c++17']
            if args.build_profile=='native' or (args.build_profile=='recorded' and name in {'static_time_scan','static_bifurcation_peaks','full_lyapunov_spectra'}):cmd+=['-march=native']
            # OpenMP is optional; serial builds retain the same per-trajectory algorithm.
            omp=[]
            if Path(compiler).name.lower().startswith(('g++','c++')):omp=['-fopenmp']
            run('compile_'+name,cmd+omp+[SOURCES[name],'-o',dest]);binaries[name]=dest
        return binaries[name]
    def wanted(name):return args.task in ('all',name)
    error=None
    try:
        if wanted('scans'):
            run('static_and_periodic_scans',[binary('static_time_scan'),pii/'riccati'])
            for f in ['time_scan_transition_band.csv','time_scan_full_range_regularized.csv']:shutil.move(pii/'riccati'/f,pii/'time_switching'/f)
        if wanted('peaks'):run('static_bifurcation_peaks',[binary('static_bifurcation_peaks'),pii/'riccati/static_bifurcation_peaks.csv'])
        if wanted('spectra'):run('full_lyapunov_spectra',[binary('full_lyapunov_spectra'),pii/'riccati/full_lyapunov_spectra.csv'])
        if wanted('dense'):run('dense_riccati',[binary('generate_dense_riccati'),pii/'riccati_dense'])
        if wanted('branches'):
            b=pii/'riccati/periodic_branches'
            for name in ['branch_continue','cascade_continue','p32_continue','window_continue']:run(name,[binary(name),b])
            recipes=json.loads((ROOT/'configs/refinement_recipes.json').read_text())['recipes'];seeds=out/'input_seeds';seeds.mkdir()
            for r in recipes:
                seed=seeds/(r['name']+'.csv')
                with seed.open('w',newline='') as f:
                    wr=csv.writer(f);wr.writerow(['gamma','T','res','it']+['x'+str(i) for i in range(5)]);wr.writerow([r['seed_gamma'],r['seed_period'],0,0]+r['seed_state'])
                run(r['name'],[binary('branch_refine'),seed,b/(r['name']+'.csv'),r['seed_gamma'],r['gamma_start'],r['gamma_end'],r['gamma_step'],r['integration_step']])
            run('summarise_branches',[sys.executable,RIC/'periodic_branches/summarise_branches.py','--data',b,'--orbits'])
        if wanted('validation'):
            v=pii/'riccati/validation'
            # These checks reintegrate the paper's supplied orbit seeds independently.
            run('adaptive_validation',[sys.executable,RIC/'validation/verify_section9.py','--extended','--output',v])
            for tag,dt,acc in [('dt005_T2000',.005,2000),('dt0025_T2000',.0025,2000),('dt0025_T4000',.0025,4000)]:
                run('targeted_'+tag,[binary('targeted_spectrum'),v/f'spectrum_gamma060_{tag}.csv',.6,dt,1000,acc])
            phase_args=['--output',v]
            if args.task=='all':phase_args+=['--package',pii.parent]
            run('phase_samples',[sys.executable,RIC/'validation/sample_archived_forcing_phase.py',*phase_args])
            data_args=['--data',pii/'riccati/periodic_branches'] if args.task=='all' else []
            run('flip_brackets',[sys.executable,RIC/'validation/localise_archived_events.py',*data_args,'--output',v/'primary_flip_brackets.csv'])
        if wanted('spin'):
            run('dense_spin',[binary('generate_dense_spin'),pi/'spin_dense'])
            run('dense_spin_checks',[sys.executable,ROOT/'tests/check_spin_output.py',pi/'spin_dense'])
        if wanted('spin-scan'):
            run('spin_lyapunov_scan',[binary('lyapunov_scan'),pi/'spin/representative_lyapunov_scan.csv'])
        if args.task=='smoke':
            run('spin_short',[binary('lyapunov_scan'),pi/'spin/representative_lyapunov_scan.csv','--transient-periods',5,'--periods',20,'--omegas','.3,1.2'])
            run('riccati_short_spectrum',[binary('targeted_spectrum'),pii/'riccati/validation/smoke_spectrum.csv',.6,.005,10,20])
            seed=ROOT/'part-II/data/riccati/periodic_branches/primary_up.csv'
            run('shoot_one_orbit',[binary('branch_refine'),seed,pii/'riccati/periodic_branches/smoke_orbit.csv',.4,.4,.4,.001,.001])
        run('check_numerical_outputs',[sys.executable,ROOT/'tests/check_numerical_outputs.py',out,'--task',args.task])
        if args.plots:
            for part,script,data in [('I',SPIN/'make_dense_figures.py',pi),('II',RIC/'make_dense_figures.py',pii)]:
                run('plot_generated_part_'+part,[sys.executable,script],{'PENTARNION_DATA_ROOT':str(data),'PENTARNION_OUTPUT_DIR':str(out/f'plots/part-{part}')})
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:error=str(exc)
    unchanged=before==reference_hashes()
    if not unchanged:error=(error+'; ' if error else '')+'Reference data changed'
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','sympy','pandas','matplotlib','mpmath']}
    manifest={'status':'PASS' if error is None else 'FAIL','task':args.task,'profile':'short test' if args.task=='smoke' else 'declared full recipes','created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'versions':versions,'compiler':compiler,'build_profile':args.build_profile,'threads':args.threads,'elapsed_seconds':round(time.monotonic()-start,3),'reference_data_unchanged':unchanged,'reference_file_count':len(before),'sources':{str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and 'outputs' not in p.parts and p.suffix in ('.cpp','.py')},'tasks':events,'error':error,'interpretation':'Executed numerical experiments and invariant/residual checks. Finite-time numerical evidence, not interval certification or proof of chaos. Reference comparisons are recorded separately.'}
    (out/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print('Report:',out)
    if error:print(error,file=sys.stderr);return 1
    print('REQUESTED NUMERICAL WORKFLOWS PASSED; reference data unchanged.');return 0

if __name__=='__main__':raise SystemExit(main())
