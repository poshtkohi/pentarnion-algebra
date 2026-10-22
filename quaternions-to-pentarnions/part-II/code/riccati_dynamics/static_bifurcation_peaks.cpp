// Dense post-transient local maxima used for the static bifurcation diagram.
// Compile: g++ -O3 -march=native -fopenmp static_bifurcation_peaks.cpp -o static_bifurcation_peaks
// Run:     ./static_bifurcation_peaks ../data/static_bifurcation_peaks.csv

#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using Vec=std::array<double,5>;
inline Vec add(const Vec&a,const Vec&b){Vec c{};for(int i=0;i<5;i++)c[i]=a[i]+b[i];return c;}
inline Vec sc(const Vec&a,double s){Vec c{};for(int i=0;i<5;i++)c[i]=a[i]*s;return c;}
inline Vec product(const Vec&a,const Vec&b,double g){
 double A=1/std::sqrt(1+g*g),B=g*A,r=1/std::sqrt(2.0);
 Vec q{a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4],
       a[0]*b[1]+a[1]*b[0],a[0]*b[2]+a[2]*b[0],
       a[0]*b[3]+a[3]*b[0],a[0]*b[4]+a[4]*b[0]};
 double s;
 s=a[1]*b[2]-a[2]*b[1];q[3]+=A*s;q[4]+=B*s;
 s=a[2]*b[3]-a[3]*b[2];q[1]+=A*s;q[4]+=B*s;
 s=a[3]*b[1]-a[1]*b[3];q[2]+=A*s;q[4]+=B*s;
 s=a[4]*b[1]-a[1]*b[4];q[2]+=r*s;q[3]+=r*s;
 s=a[4]*b[2]-a[2]*b[4];q[3]+=r*s;q[1]+=r*s;
 s=a[4]*b[3]-a[3]*b[4];q[1]+=r*s;q[2]+=r*s;return q;
}
inline Vec F(const Vec&p,double g){Vec A{-0.5,-2,0.5,-1,-0.5*g},B{1,0,2,0,-0.5*g};return add(product(A,p,g),product(B,product(p,p,g),g));}
inline Vec step(Vec y,double h,double g){
 Vec k1=F(y,g),k2=F(add(y,sc(k1,h/2)),g),k3=F(add(y,sc(k2,h/2)),g),k4=F(add(y,sc(k3,h)),g);
 for(int i=0;i<5;i++)y[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;return y;
}
struct Row{double g;std::vector<double> peaks;};
int main(int argc,char**argv){
 if(argc<2){std::cerr<<"Usage: static_bifurcation_peaks OUTPUT.csv\n";return 2;}
 std::vector<Row> rows(551);
 #pragma omp parallel for schedule(dynamic)
 for(int ii=0;ii<551;ii++){
   const double g=.35+.001*ii,dt=.01,T=1500,trans=1000;
   Vec p{.5,-.5,-.5,.5,-.5*g};
   const long N=std::llround(T/dt),tr=std::llround(trans/dt);
   double z2=p[3],z1=p[3];std::vector<double> pk;
   for(long n=1;n<=N;n++){
     p=step(p,dt,g);const double z=p[3];
     if(n>tr+2 && z1>z2 && z1>z)pk.push_back(z1);
     z2=z1;z1=z;
   }
   if(pk.size()>150)pk.erase(pk.begin(),pk.end()-150);
   rows[ii]={g,pk};
 }
 std::filesystem::create_directories(std::filesystem::path(argv[1]).parent_path());
    std::ofstream f(argv[1]); if(!f) throw std::runtime_error("Cannot open output");f<<"gamma,z_max\n"<<std::setprecision(12);
 for(const auto&r:rows)for(double z:r.peaks)f<<r.g<<','<<z<<'\n';
}
