#!/usr/bin/env python3
"""Part-aware checks for the two-paper computational companion.

quick: source layout, numerical model sanity checks and immutable data integrity.
exact: quick + selected exact algebra identities for the chosen part(s).
full: exact + the ordered-product solve and six generic Jacobian determinants.
--plots redraws available figures, primarily from supplied reference data;
--simulate-spin compiles and reruns the two supplied C++ trajectory cases.
Use run_numerics.py for the numerical scan/continuation/validation campaigns.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parent
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')


def environment() -> dict:
    out={'python':sys.version,'platform':platform.platform()}
    for package in ('numpy','scipy','sympy','pandas','matplotlib','mpmath'):
        try:out[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:out[package]='not installed'
    return out


def reference_integrity(part: str) -> int:
    inventory=json.loads((ROOT/'docs/reference_integrity.json').read_text(encoding='utf-8'))
    count=0
    for item in inventory['files']:
        rel=item['path']
        if part!='all' and '/part-'+part+'/' not in '/'+rel:continue
        path=REPOSITORY/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError(f'Reference integrity failure: {rel}')
        count+=1
    return count


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['I','II','all'],default='all')
    parser.add_argument('--level',choices=['quick','exact','full'],default='quick')
    parser.add_argument('--plots',action='store_true',help='Generate all 54 listed manuscript image inputs')
    parser.add_argument('--simulate-spin',action='store_true',help='Compile and run the supplied dense spin generator (Part I)')
    parser.add_argument('--timeout',type=float,default=1800,help='Timeout per subprocess in seconds')
    parser.add_argument('--report-dir',type=Path,help='New directory for logs and generated results')
    args=parser.parse_args()
    if args.timeout<=0:parser.error('--timeout must be positive')
    if args.simulate_spin and args.part=='II':parser.error('--simulate-spin requires Part I or all')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=(args.report_dir or ROOT/'outputs/checks'/stamp).expanduser().resolve()
    if report.exists():parser.error('--report-dir must not already exist')
    # Do not allow logs or generated results to be written into reference data.
    for part in ('I','II'):
        ref=(ROOT/f'part-{part}/data').resolve()
        if report==ref or ref in report.parents:parser.error('Report directory cannot be inside reference data')
    report.mkdir(parents=True)
    env=dict(ENV,PENTARNION_TEST_PART=args.part)
    results=[]; started=time.monotonic();failure=None

    def run(name,command,cwd,extra_env=None):
        log=report/(name+'.log')
        print(f'RUN {name}',flush=True)
        begin=time.monotonic();status='PASS';code=None;reason=None
        command=[str(v) for v in command]
        try:
            with log.open('w',encoding='utf-8') as handle:
                handle.write('COMMAND: '+' '.join(command)+'\n\n');handle.flush()
                proc=subprocess.run(command,cwd=cwd,env={**env,**(extra_env or {})},stdout=handle,stderr=subprocess.STDOUT,timeout=args.timeout)
                code=proc.returncode
                if code:raise RuntimeError(f'{name} exited with {code}; see {log}')
        except (OSError,subprocess.TimeoutExpired,RuntimeError) as exc:
            status='FAIL';reason=str(exc)
        results.append({'name':name,'status':status,'returncode':code,'elapsed_seconds':round(time.monotonic()-begin,3),'log':log.name,'error':reason})
        print(f'{status} {name} ({results[-1]["elapsed_seconds"]} s)',flush=True)
        if status!='PASS':raise RuntimeError(reason)

    try:
        before=reference_integrity(args.part)
        run('quick_tests',[sys.executable,'-m','unittest','discover','-s','tests','-t','.', '-v'],ROOT)
        if args.level!='quick':
            with tempfile.TemporaryDirectory(prefix='pentarnion_exact_') as tmp:
                work=Path(tmp)/'study'
                shutil.copytree(ROOT/'shared',work/'shared',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                tasks=[]
                for part in ('I','II'):
                    if args.part not in (part,'all'):continue
                    shutil.copytree(ROOT/f'part-{part}/code',work/f'part-{part}/code',ignore=shutil.ignore_patterns('__pycache__','*.pyc','regenerated'))
                if args.part in ('I','all'):
                    tasks.extend([
                        ('part_I_associators','part-I/code/structural_invariants/generate_algebraic_invariants.py',['--no-plots']),
                        ('part_I_rotations','part-I/code/rotational_geometry/verify_exact_results.py',[]),
                        ('part_I_nonzero_jacobians','part-I/code/bracket_geometry/compute_symbolic.py',['--nonzero-only']),
                    ])
                if args.part in ('II','all'):
                    tasks.extend([
                        ('part_II_invariant_ideal','part-II/code/exact_solutions/symbolic_verification.py',[]),
                        ('part_II_hyperbolic_plane','part-II/code/exact_solutions/verify_hyperbolic_invariant_plane.py',[]),
                        ('part_II_characteristic_polynomial','part-II/code/riccati_dynamics/validation/verify_characteristic_polynomial.py',[]),
                    ])
                if args.level=='full':
                    if args.part in ('I','all'):tasks.append(('part_I_full_jacobian_determinants','part-I/code/bracket_geometry/compute_symbolic.py',[]))
                    if args.part in ('II','all'):tasks.append(('part_II_ordered_products','part-II/code/exact_solutions/verify_ordered_product_representation.py',[]))
                for name,rel,extra in tasks:
                    run(name,[sys.executable,work/rel,*extra],work)
                for part in ('I','II'):
                    output=work/f'part-{part}/outputs'
                    if output.is_dir():shutil.copytree(output,report/f'generated-symbolic/part-{part}')
        if args.plots:
            if args.part in ('I','all'):
                dest=report/'figures/part-I';dest.mkdir(parents=True)
                # Only this step repeats the small symbolic associator calculation;
                # all spin trajectory plots read supplied reference data.
                run('part_I_associator_plot',[sys.executable,ROOT/'part-I/code/structural_invariants/generate_algebraic_invariants.py','--root',report/'plot_work/structural'],ROOT)
                shutil.copy2(report/'plot_work/structural/plots/associator_norms.png',dest/'associator_norms.png')
                run('part_I_rotation_plots',[sys.executable,ROOT/'part-I/code/rotational_geometry/build_assets.py'],ROOT,{'PENTARNION_OUTPUT_DIR':str(dest)})
                run('part_I_bracket_geometry_plots',[sys.executable,ROOT/'part-I/code/bracket_geometry/generate_figures.py','--output',dest],ROOT)
                run('part_I_spin_plots',[sys.executable,ROOT/'part-I/code/spin_dynamics/make_dense_figures.py'],ROOT,{'PENTARNION_OUTPUT_DIR':str(dest)})
            if args.part in ('II','all'):
                dest=report/'figures/part-II';dest.mkdir(parents=True)
                run('part_II_elliptic_illustration',[sys.executable,ROOT/'part-II/code/exact_solutions/generate_figures_and_data.py','--output',report/'plot_work/elliptic'],ROOT)
                for p in (report/'plot_work/elliptic/plots').glob('*.png'):shutil.copy2(p,dest/p.name)
                run('part_II_riccati_plots',[sys.executable,ROOT/'part-II/code/riccati_dynamics/make_dense_figures.py'],ROOT,{'PENTARNION_OUTPUT_DIR':str(dest)})
            figure_map=json.loads((ROOT/'docs/figure_map.json').read_text(encoding='utf-8'))
            checks=[]
            for row in figure_map:
                if args.part not in (row['part'],'all'):continue
                file=report/f'figures/part-{row["part"]}'/row['figure_file']
                exists=file.is_file()
                checks.append({**row,'file_generated':exists})
                if not exists:
                    raise RuntimeError('Expected generated figure missing: '+str(file))
            (report/'figure_results.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
        if args.simulate_spin:
            compiler=shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
            if not compiler:raise RuntimeError('A C++17 compiler was not found; install one to use --simulate-spin')
            exe=report/('generate_dense_spin.exe' if os.name=='nt' else 'generate_dense_spin')
            run('compile_spin',[compiler,'-O3','-std=c++17',ROOT/'part-I/code/spin_dynamics/generate_dense_spin.cpp','-o',exe],ROOT)
            run('simulate_spin',[exe,report/'generated-numerical/spin_dense'],ROOT)
            run('check_generated_spin',[sys.executable,ROOT/'tests/check_spin_output.py',report/'generated-numerical/spin_dense'],ROOT)
        after=reference_integrity(args.part)
        if before!=after:raise RuntimeError('Reference inventory changed during the run')
    except (OSError,ValueError,RuntimeError) as exc:
        failure=str(exc)
    try:
        reference_integrity(args.part)
        references_unchanged=True
    except (OSError,ValueError,RuntimeError):
        references_unchanged=False
    summary={'status':'PASS' if failure is None else 'FAIL','part':args.part,'level':args.level,'plots_requested':args.plots,
             'spin_simulation_requested':args.simulate_spin,'environment':environment(),'elapsed_seconds':round(time.monotonic()-started,3),
             'reference_files_unchanged':references_unchanged,'tasks':results,'error':failure,
             'scope':'Symbolic/model/data checks and requested figure generation. Numerical campaigns run separately through run_numerics.py. Not interval certification.'}
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f'Report: {report}',flush=True)
    if failure:print('FAILED: '+failure,file=sys.stderr);return 1
    print('REQUESTED CHECKS PASSED. Reference data were not modified.')
    print('See run_numerics.py and each paper reproduction guide for numerical campaigns.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
