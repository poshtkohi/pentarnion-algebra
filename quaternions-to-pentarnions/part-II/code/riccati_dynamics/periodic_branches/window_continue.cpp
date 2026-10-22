#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <array>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

using Vec = std::array<double,5>;
using Mat = std::array<double,25>;

inline Vec vadd(const Vec&a,const Vec&b){Vec c;for(int i=0;i<5;i++)c[i]=a[i]+b[i];return c;}
inline Vec vsub(const Vec&a,const Vec&b){Vec c;for(int i=0;i<5;i++)c[i]=a[i]-b[i];return c;}
inline Vec vscale(const Vec&a,double s){Vec c;for(int i=0;i<5;i++)c[i]=s*a[i];return c;}
inline double dot(const Vec&a,const Vec&b){double s=0;for(int i=0;i<5;i++)s+=a[i]*b[i];return s;}
inline double norm(const Vec&a){return std::sqrt(dot(a,a));}

inline Vec prod(const Vec&a,const Vec&b,double g){
    double al=1.0/std::sqrt(1.0+g*g), be=g*al, ir2=1.0/std::sqrt(2.0);
    Vec q{};
    q[0]=a[0]*b[0]-(a[1]*b[1]+a[2]*b[2]+a[3]*b[3]+a[4]*b[4]);
    for(int i=1;i<5;i++)q[i]=a[0]*b[i]+b[0]*a[i];
    double s;
    s=a[1]*b[2]-a[2]*b[1]; q[3]+=al*s; q[4]+=be*s;
    s=a[2]*b[3]-a[3]*b[2]; q[1]+=al*s; q[4]+=be*s;
    s=a[3]*b[1]-a[1]*b[3]; q[2]+=al*s; q[4]+=be*s;
    s=a[4]*b[1]-a[1]*b[4]; q[2]+=ir2*s; q[3]+=ir2*s;
    s=a[4]*b[2]-a[2]*b[4]; q[3]+=ir2*s; q[1]+=ir2*s;
    s=a[4]*b[3]-a[3]*b[4]; q[1]+=ir2*s; q[2]+=ir2*s;
    return q;
}
inline Vec F(const Vec&p,double g){
    Vec A{-0.5,-2.0,0.5,-1.0,-0.5*g}, B{1.0,0.0,2.0,0.0,-0.5*g};
    return vadd(prod(A,p,g),prod(B,prod(p,p,g),g));
}
inline Vec Jact(const Vec&p,const Vec&h,double g){
    Vec A{-0.5,-2.0,0.5,-1.0,-0.5*g}, B{1.0,0.0,2.0,0.0,-0.5*g};
    return vadd(prod(A,h,g),prod(B,vadd(prod(p,h,g),prod(h,p,g)),g));
}
struct Aug{Vec p;Mat M;};
inline Aug deriv(const Aug&Y,double g){
    Aug D;D.p=F(Y.p,g);
    for(int c=0;c<5;c++){
        Vec h;for(int r=0;r<5;r++)h[r]=Y.M[r*5+c];
        Vec jh=Jact(Y.p,h,g);
        for(int r=0;r<5;r++)D.M[r*5+c]=jh[r];
    }
    return D;
}
inline Aug axpy(const Aug&A,const Aug&B,double s){
    Aug C;for(int i=0;i<5;i++)C.p[i]=A.p[i]+s*B.p[i];
    for(int i=0;i<25;i++)C.M[i]=A.M[i]+s*B.M[i];return C;
}
inline void rk4(Aug&Y,double h,double g){
    Aug k1=deriv(Y,g),k2=deriv(axpy(Y,k1,h/2),g),k3=deriv(axpy(Y,k2,h/2),g),k4=deriv(axpy(Y,k3,h),g);
    for(int i=0;i<5;i++)Y.p[i]+=h*(k1.p[i]+2*k2.p[i]+2*k3.p[i]+k4.p[i])/6;
    for(int i=0;i<25;i++)Y.M[i]+=h*(k1.M[i]+2*k2.M[i]+2*k3.M[i]+k4.M[i])/6;
}
inline Aug integrate(const Vec&x,double T,double g,double dt){
    if(!(T>0) || !(dt>0) || !std::isfinite(T) || T/dt>20000000)throw std::runtime_error("Invalid integration period or step count");
    int n=std::max(20,(int)std::ceil(T/dt)); double h=T/n;
    Aug Y;Y.p=x;Y.M.fill(0);for(int i=0;i<5;i++)Y.M[i*5+i]=1;
    for(int i=0;i<n;i++)rk4(Y,h,g);
    return Y;
}

bool solve6(double A[6][6],double b[6],double x[6]){
    double aug[6][7];for(int i=0;i<6;i++){for(int j=0;j<6;j++)aug[i][j]=A[i][j];aug[i][6]=b[i];}
    for(int k=0;k<6;k++){
        int p=k;for(int i=k+1;i<6;i++)if(std::abs(aug[i][k])>std::abs(aug[p][k]))p=i;
        if(std::abs(aug[p][k])<1e-14)return false;
        if(p!=k)for(int j=k;j<7;j++)std::swap(aug[p][j],aug[k][j]);
        double piv=aug[k][k];for(int j=k;j<7;j++)aug[k][j]/=piv;
        for(int i=0;i<6;i++)if(i!=k){double m=aug[i][k];for(int j=k;j<7;j++)aug[i][j]-=m*aug[k][j];}
    }
    for(int i=0;i<6;i++)x[i]=aug[i][6];return true;
}

struct ShootResult{Vec x;double T,res;Mat M;int it;bool ok;};
ShootResult shoot(Vec x,double T,double g,double dt=0.001,int maxit=15,double tol=1e-10){
    Vec xref=x, fref=F(xref,g); Aug last;
    double res=1e300;int it=0;
    for(it=0;it<maxit;it++){
        last=integrate(x,T,g,dt);
        Vec R5=vsub(last.p,x); double phase=dot(vsub(x,xref),fref);
        res=std::sqrt(dot(R5,R5)+phase*phase);
        if(res<tol)return {x,T,res,last.M,it,true};
        double A6[6][6]{}; double rhs[6],del[6];
        Vec fend=F(last.p,g);
        for(int i=0;i<5;i++){
            rhs[i]=-R5[i];
            for(int j=0;j<5;j++)A6[i][j]=last.M[i*5+j]-(i==j?1.0:0.0);
            A6[i][5]=fend[i];
        }
        rhs[5]=-phase;for(int j=0;j<5;j++)A6[5][j]=fref[j];A6[5][5]=0;
        if(!solve6(A6,rhs,del))return {x,T,res,last.M,it,false};
        double lam=1.0; bool accepted=false;
        for(int ls=0;ls<8;ls++){
            Vec xt=x;for(int i=0;i<5;i++)xt[i]+=lam*del[i];double Tt=T+lam*del[5];
            if(Tt<=0){lam*=0.5;continue;}
            Aug test=integrate(xt,Tt,g,dt);
            Vec rr=vsub(test.p,xt);double ph=dot(vsub(xt,xref),fref);
            double nr=std::sqrt(dot(rr,rr)+ph*ph);
            if(nr<res){x=xt;T=Tt;accepted=true;break;}
            lam*=0.5;
        }
        if(!accepted){for(int i=0;i<5;i++)x[i]+=0.1*del[i];T+=0.1*del[5];}
    }
    last=integrate(x,T,g,dt);Vec R5=vsub(last.p,x);double phase=dot(vsub(x,xref),fref);
    res=std::sqrt(dot(R5,R5)+phase*phase);
    return {x,T,res,last.M,it,res<1e-7};
}

void write_header(std::ofstream&f){
    f<<"gamma,T,res,it,x0,x1,x2,x3,x4";
    for(int i=0;i<25;i++)f<<",m"<<i;
    f<<"\n";
}
void write_row(std::ofstream&f,double g,const ShootResult&r){
    f<<std::setprecision(16)<<g<<","<<r.T<<","<<r.res<<","<<r.it;
    for(double v:r.x)f<<","<<v;for(double v:r.M)f<<","<<v;f<<"\n";
}
void continue_natural(const std::string&path,Vec x,double T,double g0,double g1,double dg,double dt){
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot open output: "+path);write_header(f);
    int N=(int)std::floor(std::abs((g1-g0)/dg)+0.5)+1;
    double sign=(g1>=g0)?1:-1;double step=std::abs(dg)*sign;
    for(int k=0;k<N;k++){
        double g=g0+k*step;
        ShootResult r=shoot(x,T,g,dt);
        std::cerr<<path<<" g="<<g<<" T="<<r.T<<" res="<<r.res<<" ok="<<r.ok<<"\n";
        if(!r.ok)break;
        write_row(f,g,r);x=r.x;T=r.T;
    }
}

inline Vec rk4state(Vec y,double h,double g){
    Vec k1=F(y,g),k2=F(vadd(y,vscale(k1,h/2)),g),k3=F(vadd(y,vscale(k2,h/2)),g),k4=F(vadd(y,vscale(k3,h)),g);
    for(int i=0;i<5;i++)y[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;return y;
}
Vec settle(double g,double T,double dt){Vec p{.5,-.5,-.5,.5,-.5*g};int N=std::llround(T/dt);for(int i=0;i<N;i++)p=rk4state(p,dt,g);return p;}
int main(int argc,char**argv){if(argc!=2){std::cerr<<"Usage: window_continue OUTPUT_DIRECTORY\n";return 2;} std::string out=argv[1];std::filesystem::create_directories(out);
 struct S{const char*name;double g,T,tr,lo,hi,dg,dt;};
 std::vector<S> v{
  {"window_C1",.74,5.69998865026243,3500,.70,.7792,.0002,.001},
  {"window_C2",.78,11.3419399607,6000,.7787,.7932,.0001,.001},
  {"window_C4",.795,22.7320093212,12000,.7930,.7956,.00005,.001}
 };
 for(auto&s:v){Vec x=settle(s.g,s.tr,.005);auto r=shoot(x,s.T,s.g,s.dt,30,1e-11);std::cerr<<s.name<<" seed res="<<r.res<<" T="<<r.T<<" ok="<<r.ok<<" x=";for(double z:r.x)std::cerr<<z<<" ";std::cerr<<"\n";if(!r.ok)continue;continue_natural(out+"/"+s.name+"_down.csv",r.x,r.T,s.g,s.lo,s.dg,s.dt);continue_natural(out+"/"+s.name+"_up.csv",r.x,r.T,s.g,s.hi,s.dg,s.dt);}
}
