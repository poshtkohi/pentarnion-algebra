"""Exact symbolic certificate for the switched-algebra Jacobi family.

The script verifies that the quadratic part of the discovered component field
can be written as p star_gamma A(p) + B(p) star_gamma p for suitable real-linear
maps A and B.  The trigonometric rationalisation
    gamma = 2*t/(1-t**2)
is used, so all structure constants are rational functions of t (and sqrt(2)).

It also verifies the invariant-ideal Lie derivative identities for the general
(m, q)-family.  Run with SymPy >= 1.12.
"""
import sympy as s

t,m,q,mu=s.symbols('t m q mu', nonzero=True)
gamma=2*t/(1-t**2)
alpha=(1-t**2)/(1+t**2)
beta=2*t/(1+t**2)
rt2=s.sqrt(2)

# multiplication tensor in basis (1,i,j,k,l)
T=[[[s.Integer(0) for _ in range(5)] for _ in range(5)] for _ in range(5)]
for a in range(5):
    T[0][a][a]=1; T[a][0][a]=1
for a in range(1,5): T[a][a][0]=-1

def setv(a,b,d):
    for k,v in d.items(): T[a][b][k]=v
setv(1,2,{3:alpha,4:beta}); setv(2,1,{3:-alpha,4:-beta})
setv(2,3,{1:alpha,4:beta}); setv(3,2,{1:-alpha,4:-beta})
setv(3,1,{2:alpha,4:beta}); setv(1,3,{2:-alpha,4:-beta})
setv(4,1,{2:1/rt2,3:1/rt2}); setv(1,4,{2:-1/rt2,3:-1/rt2})
setv(4,2,{3:1/rt2,1:1/rt2}); setv(2,4,{3:-1/rt2,1:-1/rt2})
setv(4,3,{1:1/rt2,2:1/rt2}); setv(3,4,{1:-1/rt2,2:-1/rt2})

mons=[(a,b) for a in range(5) for b in range(a,5)]
Avec=s.symbols('A0:25'); Bvec=s.symbols('B0:25')
unknowns=list(Avec)+list(Bvec)
def A(r,c): return Avec[5*r+c]
def B(r,c): return Bvec[5*r+c]

# Target homogeneous quadratic field.
D={}
def put(k,a,b,v):
    if a>b: a,b=b,a
    D[(k,a,b)] = D.get((k,a,b),0)+v
put(0,0,1,1); put(0,0,3,2*(1+m)); put(0,1,2,-2*(1+m))
put(1,0,0,-1); put(1,0,2,2*m)
put(2,0,1,2)
put(3,0,0,-2*(1+m)); put(3,0,4,3*m/q)
put(4,0,3,3*q)

rows=[]; rhs=[]
for k in range(5):
    for a,b in mons:
        expr=0
        for r in range(5):
            if a==b:
                expr += A(r,b)*T[a][r][k]
            else:
                expr += A(r,b)*T[a][r][k] + A(r,a)*T[b][r][k]
        for r in range(5):
            if a==b:
                expr += B(r,a)*T[r][b][k]
            else:
                expr += B(r,a)*T[r][b][k] + B(r,b)*T[r][a][k]
        rows.append([s.expand(expr).coeff(u) for u in unknowns])
        rhs.append(D.get((k,a,b),0))
M=s.Matrix(rows); target=s.Matrix(rhs)
solution_set=s.linsolve((M,target), unknowns)
if solution_set is s.EmptySet:
    raise RuntimeError('Ordered-product coefficient system is inconsistent')
sol_tuple=next(iter(solution_set))
# Select a simple gauge by setting all free parameters to zero.
free=set().union(*(expr.free_symbols for expr in sol_tuple)) & set(unknowns)
gauge={u:s.Integer(0) for u in free}
sol=s.Matrix([s.simplify(expr.subs(gauge)) for expr in sol_tuple])
residual=[s.factor(x) for x in (M*sol-target)]
if not all(x==0 for x in residual):
    raise RuntimeError('Nonzero general coefficient residual')

# Verify the nonsingular flagship path q=gamma, m=mu*gamma.
path_sub={q:gamma,m:mu*gamma}
path_sol=s.Matrix([s.factor(expr.subs(path_sub)) for expr in sol])
path_residual=[s.factor(x.subs(path_sub)) for x in (M*sol-target)]
if not all(x==0 for x in path_residual):
    raise RuntimeError('Nonzero switching-path coefficient residual')

# Invariant ideal verification for the general (m,q) component field.
w,x,y,z,u=s.symbols('w x y z u')
F=s.Matrix([
    x-z+w*x+2*(1+m)*w*z-2*(1+m)*x*y,
    y-(1+m)*w-w**2+2*m*w*y,
    2*w*x,
    1-2*(1+m)*w**2+(3*m/q)*u*w,
    3*q*w*z,
])
G=[y-w**2, z-w*x, u-q*w*y, x**2-m*y**2+(1+m)*y-1]
vars=(w,x,y,z,u)
LF=[s.expand(sum(s.diff(g,v)*F[j] for j,v in enumerate(vars))) for g in G]
claimed=[
    4*(1+m)*w*x*G[0] + 2*w*(1-2*(1+m)*w)*G[1],
    (2*(1+m)*x**2-m*y-w+1+m)*G[0]
      + x*(1-2*(1+m)*w)*G[1] + (3*m*w/q)*G[2] - G[3],
    q*x*(2*(1+m)*y-1)*G[0] + q*(y+3*w-2*(1+m)*w*y)*G[1],
    2*x*G[0],
]
if not all(s.expand(a-b)==0 for a,b in zip(LF,claimed)):
    raise RuntimeError('Invariant-ideal identity failed')

# Write one exact matrix representative for inspection.
Amat=s.Matrix(5,5,path_sol[:25])
Bmat=s.Matrix(5,5,path_sol[25:])
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / 'outputs/exact_integrable/ordered_product_maps_flagship.txt'
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8') as f:
    f.write('Rationalisation: gamma = 2*t/(1-t**2), m=mu*gamma, q=gamma\n\n')
    f.write('A(t,mu) =\n'+s.pretty(Amat)+'\n\n')
    f.write('B(t,mu) =\n'+s.pretty(Bmat)+'\n')

print('PASS: 75 quadratic coefficient equations verified exactly.')
print('PASS: invariant-ideal Lie derivative identities verified exactly.')
print('Free parameters fixed by gauge:', len(free))

# Added for the manuscript's uniform-existence discussion. This verifies that
# every denominator has only the displayed nonvanishing factors. The exact
# six-dimensional homogeneous-kernel argument remains an analytical proof in
# the paper; a rational-function solve alone proves only generic nullity.
allowed = [t-1,t+1,t*t+1,t**4+26*t*t+1]
found = set()
for expr in path_sol:
    den=s.denom(s.cancel(expr,extension=rt2))
    factors=s.factor_list(den,t,extension=rt2)[1]
    for factor,power in factors:
        factor=s.Poly(factor,t,extension=rt2).monic().as_expr()
        if not any(s.expand(factor-s.Poly(g,t,extension=rt2).monic().as_expr())==0 for g in allowed):
            raise RuntimeError(f'Unrecognised parameter denominator factor: {factor}')
        found.add(s.sstr(factor))
if len(free)!=6:
    raise RuntimeError(f'Expected generic nullity six, found {len(free)}')
import json
report={
    'shape':list(M.shape),
    'generic_nullity':len(free),
    'generic_rank':50-len(free),
    'coefficient_residuals_zero':True,
    'path_coefficient_residuals_zero':True,
    'parameter_domain':'0 < t <= sqrt(2)-1, 0 < mu < 1',
    'denominator_factors':sorted(found),
    'nonvanishing_reason':'t<1; t+1>0; t^2+1>0; t^4+26t^2+1>0',
    'rank_scope':'Generic symbolic rank; uniform kernel dimension is proved analytically in the manuscript.',
    'A_entries':[s.sstr(v) for v in Amat],
    'B_entries':[s.sstr(v) for v in Bmat],
    'entry_order':'row-major; columns and rows use (1,i,j,k,ell)',
}
(OUT.parent/'ordered_product_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('PASS: all parameter denominator factors are nonvanishing on the stated path.')
