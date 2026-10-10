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
assert solution_set is not s.EmptySet
sol_tuple=next(iter(solution_set))
# Select a simple gauge by setting all free parameters to zero.
free=set().union(*(expr.free_symbols for expr in sol_tuple)) & set(unknowns)
gauge={u:s.Integer(0) for u in free}
sol=s.Matrix([s.simplify(expr.subs(gauge)) for expr in sol_tuple])
residual=[s.factor(x) for x in (M*sol-target)]
assert all(x==0 for x in residual)

# Verify the nonsingular flagship path q=gamma, m=mu*gamma.
path_sub={q:gamma,m:mu*gamma}
path_sol=s.Matrix([s.factor(expr.subs(path_sub)) for expr in sol])
path_residual=[s.factor(x.subs(path_sub)) for x in (M*sol-target)]
assert all(x==0 for x in path_residual)

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
assert all(s.expand(a-b)==0 for a,b in zip(LF,claimed))

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
