// Targeted variational Lyapunov-spectrum checks for the Section 9 revision.
// No changes to the archived scans. These are finite-time, non-validated diagnostics.
// c++ -O3 -std=c++17 targeted_spectrum.cpp -o targeted_spectrum
// ./targeted_spectrum output.csv gamma dt transient accumulation
#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
using Vec=std::array<double,5>;
using Mat=std::array<double,25>;
struct Aug {Vec p{};Mat M{};double integral_divergence=0;};
Vec product(const Vec&a,const Vec&b,double g){
 const double k=1/std::sqrt(1+g*g),s=g*k,r=1/std::sqrt(2.0);
 Vec q{};q[0]=a[0]*b[0];for(int i=1;i<5;++i){q[0]-=a[i]*b[i];q[i]=a[0]*b[i]+a[i]*b[0];}
 for(auto ijk: {std::array<int,3>{1,2,3},std::array<int,3>{2,3,1},std::array<int,3>{3,1,2}}){
  const int i=ijk[0],j=ijk[1],l=ijk[2];double z=a[i]*b[j]-a[j]*b[i];q[l]+=k*z;q[4]+=s*z;
  z=a[4]*b[i]-a[i]*b[4];q[j]+=r*z;q[l]+=r*z;
 }return q;
}
Vec add(Vec a,const Vec&b){for(int i=0;i<5;++i)a[i]+=b[i];return a;}
Aug derivative(const Aug&y,double g){
 const Vec A{-.5,-2,.5,-1,-.5*g},B{1,0,2,0,-.5*g};Aug d;
 d.p=add(product(A,y.p,g),product(B,product(y.p,y.p,g),g));
 for(int c=0;c<5;++c){Vec h;for(int r=0;r<5;++r)h[r]=y.M[5*r+c];
  Vec z=add(product(A,h,g),product(B,add(product(y.p,h,g),product(h,y.p,g)),g));
  for(int r=0;r<5;++r)d.M[5*r+c]=z[r];
 }d.integral_divergence=2*g*y.p[4]+10*y.p[0]-8*y.p[2]-2.5;return d;
}
Aug axpy(Aug a,const Aug&b,double h){for(int i=0;i<5;++i)a.p[i]+=h*b.p[i];for(int i=0;i<25;++i)a.M[i]+=h*b.M[i];a.integral_divergence+=h*b.integral_divergence;return a;}
void step(Aug&y,double h,double g){
 const auto k1=derivative(y,g),k2=derivative(axpy(y,k1,h/2),g),k3=derivative(axpy(y,k2,h/2),g),k4=derivative(axpy(y,k3,h),g);
 for(int i=0;i<5;++i)y.p[i]+=h*(k1.p[i]+2*k2.p[i]+2*k3.p[i]+k4.p[i])/6;
 for(int i=0;i<25;++i)y.M[i]+=h*(k1.M[i]+2*k2.M[i]+2*k3.M[i]+k4.M[i])/6;
 y.integral_divergence+=h*(k1.integral_divergence+2*k2.integral_divergence+2*k3.integral_divergence+k4.integral_divergence)/6;
 for(double z:y.p)if(!std::isfinite(z))throw std::runtime_error("Non-finite state");
}
Vec qr(Mat&m){
 Vec logs{};
 // Reorthogonalised modified Gram--Schmidt; diagonal norms are positive.
 for(int j=0;j<5;++j){for(int pass=0;pass<2;++pass)for(int k=0;k<j;++k){
  double dot=0;for(int r=0;r<5;++r)dot+=m[5*r+j]*m[5*r+k];
  for(int r=0;r<5;++r)m[5*r+j]-=dot*m[5*r+k];
 }
 double norm=0;for(int r=0;r<5;++r)norm+=m[5*r+j]*m[5*r+j];norm=std::sqrt(norm);
 if(!(norm>0) || !std::isfinite(norm))throw std::runtime_error("Degenerate tangent basis");
 logs[j]=std::log(norm);for(int r=0;r<5;++r)m[5*r+j]/=norm;
 }return logs;
}
int main(int argc,char**argv){try{
 if(argc!=6){std::cerr<<"Usage: targeted_spectrum output.csv gamma dt transient accumulation\n";return 2;}
 const double g=std::stod(argv[2]),dt=std::stod(argv[3]),trans=std::stod(argv[4]),acc=std::stod(argv[5]),qrtime=.1;
 if(g<0 || g>1 || !(dt>0) || trans<0 || !(acc>0))throw std::invalid_argument("Invalid parameters");
 const long nq=std::llround(qrtime/dt),nt=std::llround(trans/dt),na=std::llround(acc/dt);
 if(nq<1 || nt%nq || na%nq || std::abs(nq*dt-qrtime)>1e-10 || std::abs(nt*dt-trans)>1e-9 || std::abs(na*dt-acc)>1e-9)throw std::invalid_argument("Time intervals must align with dt and QR interval");
 Aug y;y.p={.5,-.5,-.5,.5,-.5*g};for(int i=0;i<5;++i)y.M[5*i+i]=1;
 Vec sums{};double baseline=0,maxradius=0;
 for(long n=1;n<=nt+na;++n){step(y,dt,g);if(n==nt)baseline=y.integral_divergence;
  if(n%nq==0){auto logs=qr(y.M);if(n>nt)for(int j=0;j<5;++j)sums[j]+=logs[j];}
  if(n>nt){double r=0;for(double z:y.p)r+=z*z;maxradius=std::max(maxradius,std::sqrt(r));}
 }
 for(double&v:sums)v/=acc;std::sort(sums.begin(),sums.end(),std::greater<double>());
 double total=0;for(double v:sums)total+=v;double mean_div=(y.integral_divergence-baseline)/acc;
 std::filesystem::create_directories(std::filesystem::path(argv[1]).parent_path());
 std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot open output");
 out<<"gamma,dt,transient,accumulation,qr_interval,lambda1,lambda2,lambda3,lambda4,lambda5,mean_divergence,sum_minus_mean_divergence,max_sampled_radius\n"<<std::setprecision(16);
 out<<g<<','<<dt<<','<<trans<<','<<acc<<','<<qrtime;for(double v:sums)out<<','<<v;
 out<<','<<mean_div<<','<<total-mean_div<<','<<maxradius<<'\n';
 std::cout<<"gamma="<<g<<" dt="<<dt<<" accumulation="<<acc<<" lambda1="<<sums[0]<<" trace_check="<<total-mean_div<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
