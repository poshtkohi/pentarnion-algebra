#!/usr/bin/env python3
"""Cross-check C++ fields against the independent Python coordinate model."""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RIC=ROOT/'part-II/code/riccati_dynamics'
sys.path.insert(0,str(RIC));from switching_algebra_model import static_rhs,tangent_action


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',default=shutil.which('g++') or shutil.which('clang++'));a=p.parse_args()
    if not a.compiler:p.error('A C++17 compiler is required')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='pentarnion_native_models_') as td:
        work=Path(td)
        for src in sorted(RIC.rglob('*.cpp')):
            if src.name=='targeted_spectrum.cpp':field='Aug y; y.p=p; return derivative(y,g).p;'
            elif src.name in ['static_time_scan.cpp','generate_dense_riccati.cpp']:field='return static_field(p,g);'
            elif src.name=='full_lyapunov_spectra.cpp':field='return field(p,g);'
            else:field='return F(p,g);'
            text='#define main numerical_driver_main\n#include "'+src.as_posix()+'"\n#undef main\n#include <iomanip>\nVec testfield(const Vec&p,double g){'+field+'}\nint main(){std::cout<<std::setprecision(17);for(double g:{0.,.2,.6,1.})for(int k=0;k<5;++k){Vec p{.13+k*.07,-.31+k*.04,.23-k*.03,.17+k*.02,-.12+k*.015};auto z=testfield(p,g);std::cout<<g<<","<<k;for(double x:z)std::cout<<","<<x;std::cout<<"\\n";}}\n'
            source=work/(src.stem+'.cpp');source.write_text(text);exe=work/(src.stem+('.exe' if sys.platform=='win32' else ''))
            subprocess.run([a.compiler,'-O2','-std=c++17',str(source),'-o',str(exe)],check=True,capture_output=True)
            output=subprocess.check_output([str(exe)],text=True);err=0.
            for line in output.splitlines():
                vals=list(map(float,line.split(',')));g,k=vals[:2];pp=np.array([.13+k*.07,-.31+k*.04,.23-k*.03,.17+k*.02,-.12+k*.015]);err=max(err,float(np.max(abs(np.array(vals[2:])-static_rhs(0,pp,g)))))
            if err>2e-13:raise AssertionError((src,err))
            rows.append({'source':str(src.relative_to(ROOT)),'sample_count':20,'max_field_error':err});print('PASS',src.name,err)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'PASS','checks':rows},indent=2)+'\n')

if __name__=='__main__':main()
