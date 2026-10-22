// Independent tangent-space Lyapunov scan of the Part I spin equations.
// The normalized RK4 map and its derivative are advanced consistently.
// Defaults: 500 transient periods, 3000 accumulation periods, dt <= 0.01.
// Reference table is an input for comparison only, never overwritten.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using V=std::array<double,3>;
using Y=std::array<double,9>;
const double PI=std::acos(-1.0);
double dot(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
V unit(V v){double n=std::sqrt(dot(v,v));if(!(n>0)||!std::isfinite(n))throw std::runtime_error("Degenerate vector");for(double&x:v)x/=n;return v;}
Y deriv(double t,const Y&y,double omega){
 const double g=.5*(1+std::sin(omega*t)),g2=g*g;
 const double A=3*(g2*g2+2*g2+3)/((1+g2)*(3+g2)),B=(4*g2+3)/(1+g2);
 const double g1=1/std::sqrt(3+g2),g2f=1/std::sqrt(2.0),g3=1/std::sqrt(6.0);
 Y z{};z[0]=-g3*y[1]+g2f*y[2];z[1]=(A-B)*y[0]*y[2]-g1*y[2]+g3*y[0];z[2]=(B-A)*y[0]*y[1]-g2f*y[0]+g1*y[1];
 for(int c=0;c<2;++c){int j=3+3*c;z[j]=-g3*y[j+1]+g2f*y[j+2];z[j+1]=((A-B)*y[2]+g3)*y[j]+((A-B)*y[0]-g1)*y[j+2];z[j+2]=((B-A)*y[1]-g2f)*y[j]+((B-A)*y[0]+g1)*y[j+1];}
 return z;
}
Y axpy(Y a,const Y&b,double h){for(int i=0;i<9;++i)a[i]+=h*b[i];return a;}
void step(Y&y,double t,double h,double omega){
 const Y k1=deriv(t,y,omega),k2=deriv(t+h/2,axpy(y,k1,h/2),omega),k3=deriv(t+h/2,axpy(y,k2,h/2),omega),k4=deriv(t+h,axpy(y,k3,h),omega);
 for(int i=0;i<9;++i)y[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
 const double n=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);if(!(n>0)||!std::isfinite(n))throw std::runtime_error("Invalid spin state");
 for(int i=0;i<3;++i)y[i]/=n;
 // Differential of y_raw -> y_raw / ||y_raw||.
 for(int c=0;c<2;++c){int j=3+3*c;double d=0;for(int i=0;i<3;++i)d+=y[i]*y[j+i];for(int i=0;i<3;++i)y[j+i]=(y[j+i]-d*y[i])/n;}
}
std::array<double,2> qr(Y&y){
 std::array<double,2> logs{};
 for(int c=0;c<2;++c){int j=3+3*c;
  for(int pass=0;pass<2;++pass){double d=0;for(int i=0;i<3;++i)d+=y[i]*y[j+i];for(int i=0;i<3;++i)y[j+i]-=d*y[i];
   if(c){d=0;for(int i=0;i<3;++i)d+=y[3+i]*y[j+i];for(int i=0;i<3;++i)y[j+i]-=d*y[3+i];}}
  double n=0;for(int i=0;i<3;++i)n+=y[j+i]*y[j+i];n=std::sqrt(n);if(!(n>0)||!std::isfinite(n))throw std::runtime_error("Degenerate tangent basis");
  logs[c]=std::log(n);for(int i=0;i<3;++i)y[j+i]/=n;
 }return logs;
}
struct Result{double omega,l1,l2,dt,norm_error;long long steps;};
Result scan(double omega,int trans,int periods,double maxdt,int qrsteps){
 const double T=2*PI/omega;const long long nperiod=std::max(20LL,(long long)std::ceil(T/maxdt));const double h=T/nperiod;
 V y0=unit({.45,.65,-.61}),e=unit(cross(y0,{1,0,0})),f=cross(y0,e);Y y{};
 for(int i=0;i<3;++i){y[i]=y0[i];y[3+i]=e[i];y[6+i]=f[i];}
 const long long nt=nperiod*trans,na=nperiod*periods;std::array<double,2>s{};double err=0;
 for(long long n=1;n<=nt+na;++n){step(y,(n-1)*h,h,omega);
  if(n%qrsteps==0 || n==nt || n==nt+na){auto l=qr(y);if(n>nt)for(int c=0;c<2;++c)s[c]+=l[c];}
  err=std::max(err,std::abs(std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2])-1));}
 s[0]/=(periods*T);s[1]/=(periods*T);if(s[1]>s[0])std::swap(s[0],s[1]);return{omega,s[0],s[1],h,err,nperiod};
}
int main(int argc,char**argv){try{
 if(argc<2){std::cerr<<"Usage: lyapunov_scan OUTPUT.csv [--transient-periods N] [--periods N] [--max-dt H] [--qr-steps N] [--omegas list]\n";return 2;}
 int trans=500,periods=3000,qrsteps=10;double maxdt=.01;std::vector<double> omegas{.1,.3,.55,.6,.7,.8,1,1.2,1.35,1.5,1.65,2,4,6};
 for(int i=2;i<argc;++i){std::string a=argv[i];if(i+1>=argc)throw std::invalid_argument("Missing option value");std::string v=argv[++i];if(a=="--transient-periods")trans=std::stoi(v);else if(a=="--periods")periods=std::stoi(v);else if(a=="--max-dt")maxdt=std::stod(v);else if(a=="--qr-steps")qrsteps=std::stoi(v);else if(a=="--omegas"){omegas.clear();std::stringstream s(v);std::string q;while(std::getline(s,q,','))omegas.push_back(std::stod(q));}else throw std::invalid_argument("Unknown option: "+a);}
 if(trans<0||periods<=0||!(maxdt>0)||!std::isfinite(maxdt)||qrsteps<1||omegas.empty())throw std::invalid_argument("Invalid settings");for(double w:omegas)if(!(w>0)||!std::isfinite(w))throw std::invalid_argument("Frequencies must be finite and positive");
 std::filesystem::path out=argv[1];if(!out.parent_path().empty())std::filesystem::create_directories(out.parent_path());
 std::vector<Result> rows(omegas.size());
 #pragma omp parallel for schedule(dynamic)
 for(int i=0;i<(int)omegas.size();++i)rows[i]=scan(omegas[i],trans,periods,maxdt,qrsteps);
 std::ofstream file(out);if(!file)throw std::runtime_error("Cannot open output file");
 file<<"Omega,lambda_max,lambda_min,sum_exponents,dt,steps_per_period,transient_periods,accumulation_periods,accumulation_time,qr_steps,max_norm_error\n"<<std::setprecision(17);
 for(const auto&r:rows){file<<r.omega<<','<<r.l1<<','<<r.l2<<','<<r.l1+r.l2<<','<<r.dt<<','<<r.steps<<','<<trans<<','<<periods<<','<<periods*2*PI/r.omega<<','<<qrsteps<<','<<r.norm_error<<'\n';std::cout<<"Omega="<<r.omega<<" LLE="<<r.l1<<" tangent-sum="<<r.l1+r.l2<<'\n';}
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
