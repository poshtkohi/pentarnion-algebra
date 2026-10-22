// Representative full Lyapunov spectra for the final static switching model.
// Compile: g++ -O3 -march=native full_lyapunov_spectra.cpp -o full_lyapunov_spectra
// Run:     ./full_lyapunov_spectra ../data/full_lyapunov_spectra.csv

#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Vec = std::array<double,5>;
using Mat = std::array<double,25>;

inline Vec add(const Vec& a,const Vec& b){Vec c{};for(int i=0;i<5;i++)c[i]=a[i]+b[i];return c;}
inline Vec scale(const Vec& a,double s){Vec c{};for(int i=0;i<5;i++)c[i]=a[i]*s;return c;}

inline Vec product(const Vec&a,const Vec&b,double g){
    const double k=1/std::sqrt(1+g*g),s=g*k,r=1/std::sqrt(2.0);
    Vec q{a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4],
          a[0]*b[1]+a[1]*b[0],a[0]*b[2]+a[2]*b[0],
          a[0]*b[3]+a[3]*b[0],a[0]*b[4]+a[4]*b[0]};
    double x;
    x=a[1]*b[2]-a[2]*b[1];q[3]+=k*x;q[4]+=s*x;
    x=a[2]*b[3]-a[3]*b[2];q[1]+=k*x;q[4]+=s*x;
    x=a[3]*b[1]-a[1]*b[3];q[2]+=k*x;q[4]+=s*x;
    x=a[4]*b[1]-a[1]*b[4];q[2]+=r*x;q[3]+=r*x;
    x=a[4]*b[2]-a[2]*b[4];q[3]+=r*x;q[1]+=r*x;
    x=a[4]*b[3]-a[3]*b[4];q[1]+=r*x;q[2]+=r*x;
    return q;
}
inline Vec field(const Vec&p,double g){
    Vec A{-0.5,-2,0.5,-1,-0.5*g},B{1,0,2,0,-0.5*g};
    return add(product(A,p,g),product(B,product(p,p,g),g));
}
inline Vec jacobian_action(const Vec&p,const Vec&h,double g){
    Vec A{-0.5,-2,0.5,-1,-0.5*g},B{1,0,2,0,-0.5*g};
    return add(product(A,h,g),product(B,add(product(p,h,g),product(h,p,g)),g));
}
struct Aug{Vec p;Mat M;};
inline Aug derivative(const Aug&Y,double g){
    Aug D{};D.p=field(Y.p,g);
    for(int c=0;c<5;c++){
        Vec h{};for(int r=0;r<5;r++)h[r]=Y.M[r*5+c];
        Vec jh=jacobian_action(Y.p,h,g);
        for(int r=0;r<5;r++)D.M[r*5+c]=jh[r];
    }
    return D;
}
inline Aug axpy(const Aug&A,const Aug&B,double s){
    Aug C{};for(int i=0;i<5;i++)C.p[i]=A.p[i]+s*B.p[i];
    for(int i=0;i<25;i++)C.M[i]=A.M[i]+s*B.M[i];return C;
}
inline void rk4_step(Aug&Y,double h,double g){
    Aug k1=derivative(Y,g),k2=derivative(axpy(Y,k1,h/2),g),
        k3=derivative(axpy(Y,k2,h/2),g),k4=derivative(axpy(Y,k3,h),g);
    for(int i=0;i<5;i++)Y.p[i]+=h*(k1.p[i]+2*k2.p[i]+2*k3.p[i]+k4.p[i])/6;
    for(int i=0;i<25;i++)Y.M[i]+=h*(k1.M[i]+2*k2.M[i]+2*k3.M[i]+k4.M[i])/6;
}
inline std::array<double,5> qr(Mat&M){
    std::array<double,5> logs{};
    for(int j=0;j<5;j++){
        for(int k=0;k<j;k++){
            double dot=0;for(int r=0;r<5;r++)dot+=M[r*5+k]*M[r*5+j];
            for(int r=0;r<5;r++)M[r*5+j]-=dot*M[r*5+k];
        }
        double n=0;for(int r=0;r<5;r++)n+=M[r*5+j]*M[r*5+j];n=std::sqrt(n);
        logs[j]=std::log(n);for(int r=0;r<5;r++)M[r*5+j]/=n;
    }
    return logs;
}
int main(int argc,char**argv){
    if(argc<2){std::cerr<<"Usage: full_lyapunov_spectra OUTPUT.csv\n";return 2;}
    std::vector<double> gs{0.0,0.35,0.40,0.47,0.50,0.55,0.65,0.70,0.75,0.80,0.855,1.0};
    std::filesystem::create_directories(std::filesystem::path(argv[1]).parent_path());
    std::ofstream f(argv[1]); if(!f) throw std::runtime_error("Cannot open output");f<<"gamma,lambda1,lambda2,lambda3,lambda4,lambda5\n"<<std::setprecision(12);
    const double dt=.005,T=3000,trans=1000,qr_time=.1;const int qr_steps=std::llround(qr_time/dt);
    for(double g:gs){
        Aug Y{};Y.p={0.5,-0.5,-0.5,0.5,-0.5*g};Y.M.fill(0);for(int i=0;i<5;i++)Y.M[i*5+i]=1;
        std::array<double,5>sums{};long count=0,N=std::llround(T/dt),tr=std::llround(trans/dt);
        for(long n=1;n<=N;n++){
            rk4_step(Y,dt,g);
            if(n%qr_steps==0){auto l=qr(Y.M);if(n>=tr){for(int j=0;j<5;j++)sums[j]+=l[j];++count;}}
        }
        std::array<double,5> ex{};for(int j=0;j<5;j++)ex[j]=sums[j]/(count*qr_time);
        std::sort(ex.begin(),ex.end(),std::greater<double>());
        f<<g;for(double x:ex)f<<','<<x;f<<'\n';
    }
}
