"""Core algebra and integrators for the sparse rational Pentarnion model."""
from __future__ import annotations
import numpy as np
from numba import njit

A = np.array([-1/2, -2, 1/2, -1, -1/2], dtype=np.float64)
B = np.array([1, 0, 2, 0, -1/2], dtype=np.float64)
P0 = np.array([1/2, -1/2, -1/2, 1/2, -1/2], dtype=np.float64)
SQRT2 = np.sqrt(2.0)



def left_matrix(q: np.ndarray) -> np.ndarray:
    """Matrix L(q) satisfying [q p]_B = L(q)[p]_B."""
    q0,q1,q2,q3,q4 = np.asarray(q, dtype=np.float64)
    s = SQRT2
    return np.array([
        [q0,-q1,-q2,-q3,-q4],
        [q1,q0,(-q3+q4)/s,(q2+q4)/s,(-q2-q3)/s],
        [q2,(q3+q4)/s,q0,(-q1+q4)/s,(-q1-q3)/s],
        [q3,(-q2+q4)/s,(q1+q4)/s,q0,(-q1-q2)/s],
        [q4,(-q2+q3)/s,(q1-q3)/s,(-q1+q2)/s,q0],
    ], dtype=np.float64)

def right_matrix(q: np.ndarray) -> np.ndarray:
    """Matrix R(q) satisfying [p q]_B = R(q)[p]_B."""
    C = np.diag([1.0,-1.0,-1.0,-1.0,-1.0])
    qbar = C @ np.asarray(q, dtype=np.float64)
    return C @ left_matrix(qbar) @ C

def square_map(x: np.ndarray) -> np.ndarray:
    """Coordinate square Q(x)=[p^2]_B."""
    x0,x1,x2,x3,x4 = np.asarray(x, dtype=np.float64)
    return np.array([x0*x0-x1*x1-x2*x2-x3*x3-x4*x4,
                     2*x0*x1,2*x0*x2,2*x0*x3,2*x0*x4], dtype=np.float64)

def rhs_matrix(x: np.ndarray, eta: float) -> np.ndarray:
    """Matrix form L(A)x+(1-eta)L(B)Q(x)+eta R(p)L(B)x."""
    x = np.asarray(x, dtype=np.float64)
    return (left_matrix(A) @ x
            + (1.0-eta)*(left_matrix(B) @ square_map(x))
            + eta*(right_matrix(x) @ (left_matrix(B) @ x)))

@njit(cache=True)
def pmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p0,p1,p2,p3,p4 = p
    q0,q1,q2,q3,q4 = q
    r = np.empty(5, dtype=np.float64)
    r[0] = p0*q0-p1*q1-p2*q2-p3*q3-p4*q4
    r[1] = p0*q1+q0*p1+(p2*q3-p2*q4-p3*q2-p3*q4+p4*q2+p4*q3)/SQRT2
    r[2] = p0*q2+q0*p2+(-p1*q3-p1*q4+p3*q1-p3*q4+p4*q1+p4*q3)/SQRT2
    r[3] = p0*q3+q0*p3+(p1*q2-p1*q4-p2*q1-p2*q4+p4*q1+p4*q2)/SQRT2
    r[4] = p0*q4+q0*p4+(p1*q2-p1*q3-p2*q1+p2*q3+p3*q1-p3*q2)/SQRT2
    return r

@njit(cache=True)
def rhs(q: np.ndarray, eta: float) -> np.ndarray:
    qq = pmul(q,q)
    return pmul(A,q) + (1.0-eta)*pmul(B,qq) + eta*pmul(pmul(B,q),q)

@njit(cache=True)
def tangent_rhs(q: np.ndarray, h: np.ndarray, eta: float) -> np.ndarray:
    left = pmul(B, pmul(q,h)+pmul(h,q))
    right = pmul(pmul(B,h),q)+pmul(pmul(B,q),h)
    return pmul(A,h)+(1.0-eta)*left+eta*right

@njit(cache=True)
def divergence(q: np.ndarray, eta: float) -> float:
    return -2.5 + 10.0*q[0] - (8.0+4.0*eta)*q[2] + (2.0+3.0*eta)*q[4]

@njit(cache=True)
def rk4_state(q: np.ndarray, eta: float, dt: float) -> np.ndarray:
    k1=rhs(q,eta); k2=rhs(q+0.5*dt*k1,eta)
    k3=rhs(q+0.5*dt*k2,eta); k4=rhs(q+dt*k3,eta)
    return q+dt*(k1+2*k2+2*k3+k4)/6.0

@njit(cache=True)
def rk4_state_tangent(q: np.ndarray, h: np.ndarray, eta: float, dt: float):
    k1q=rhs(q,eta); k1h=tangent_rhs(q,h,eta)
    q2=q+0.5*dt*k1q; h2=h+0.5*dt*k1h
    k2q=rhs(q2,eta); k2h=tangent_rhs(q2,h2,eta)
    q3=q+0.5*dt*k2q; h3=h+0.5*dt*k2h
    k3q=rhs(q3,eta); k3h=tangent_rhs(q3,h3,eta)
    q4=q+dt*k3q; h4=h+dt*k3h
    k4q=rhs(q4,eta); k4h=tangent_rhs(q4,h4,eta)
    return q+dt*(k1q+2*k2q+2*k3q+k4q)/6.0, h+dt*(k1h+2*k2h+2*k3h+k4h)/6.0

@njit(cache=True)
def _deriv_matrix(q, H, eta):
    dq=rhs(q,eta); dH=np.empty((5,5),dtype=np.float64)
    for j in range(5): dH[:,j]=tangent_rhs(q,H[:,j],eta)
    return dq,dH

@njit(cache=True)
def rk4_matrix(q,H,eta,dt):
    k1q,k1H=_deriv_matrix(q,H,eta)
    k2q,k2H=_deriv_matrix(q+0.5*dt*k1q,H+0.5*dt*k1H,eta)
    k3q,k3H=_deriv_matrix(q+0.5*dt*k2q,H+0.5*dt*k2H,eta)
    k4q,k4H=_deriv_matrix(q+dt*k3q,H+dt*k3H,eta)
    return q+dt*(k1q+2*k2q+2*k3q+k4q)/6.0, H+dt*(k1H+2*k2H+2*k3H+k4H)/6.0

@njit(cache=True)
def modified_gram_schmidt(H):
    Q=np.empty_like(H); logs=np.empty(5)
    for j in range(5):
        v=H[:,j].copy()
        for i in range(j):
            c=np.dot(Q[:,i],v); v-=c*Q[:,i]
        n=np.sqrt(np.dot(v,v)); logs[j]=np.log(n); Q[:,j]=v/n
    return Q,logs

@njit(cache=True)
def integrate(eta,dt,transient,record,stride=1):
    q=P0.copy()
    for _ in range(int(round(transient/dt))): q=rk4_state(q,eta,dt)
    n=int(round(record/dt)); m=n//stride+1
    t=np.empty(m); x=np.empty((m,5));t[0]=0.0;x[0]=q;k=1
    for j in range(1,n+1):
        q=rk4_state(q,eta,dt)
        if j%stride==0: t[k]=j*dt;x[k]=q;k+=1
    return t,x

@njit(cache=True)
def full_lyapunov(eta,dt,transient,measure,renorm=0.25,history_interval=100.0):
    q=P0.copy()
    for _ in range(int(round(transient/dt))):q=rk4_state(q,eta,dt)
    H=np.eye(5); nr=max(1,int(round(renorm/dt))); nb=int(round(measure/(nr*dt)))
    sums=np.zeros(5); divsum=0.0; ndiv=0; elapsed=0.0
    he=max(1,int(round(history_interval/(nr*dt)))); ht=np.empty(nb//he+1); hs=np.empty((nb//he+1,5));hi=0
    ht[0]=0.0;hs[0]=0.0;hi=1
    maxnorm=np.sqrt(np.dot(q,q))
    for b in range(nb):
        for _ in range(nr):
            q,H=rk4_matrix(q,H,eta,dt);divsum+=divergence(q,eta);ndiv+=1
            maxnorm=max(maxnorm,np.sqrt(np.dot(q,q)))
        H,logs=modified_gram_schmidt(H);sums+=logs;elapsed+=nr*dt
        if (b+1)%he==0:ht[hi]=elapsed;hs[hi]=sums/elapsed;hi+=1
    return sums/elapsed,divsum/ndiv,ht[:hi],hs[:hi],q,maxnorm
