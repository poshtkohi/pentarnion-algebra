#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using Vec=std::array<double,5>;
inline Vec add(const Vec&a,const Vec&b){Vec c{};for(int i=0;i<5;i++)c[i]=a[i]+b[i];return c;}
inline Vec scale(const Vec&a,double s){Vec c{};for(int i=0;i<5;i++)c[i]=s*a[i];return c;}
inline double nv(const Vec&a){double s=0;for(double v:a)s+=v*v;return std::sqrt(s);} 
inline bool ok(const Vec&a){for(double v:a)if(!std::isfinite(v))return false;return nv(a)<1e12;}
inline Vec product(const Vec&a,const Vec&b,double g){double k=1/std::sqrt(1+g*g),s=g*k,r=1/std::sqrt(2.0);Vec q{};q[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4];for(int i=1;i<5;i++)q[i]=a[0]*b[i]+a[i]*b[0];double z=a[1]*b[2]-a[2]*b[1];q[3]+=k*z;q[4]+=s*z;z=a[2]*b[3]-a[3]*b[2];q[1]+=k*z;q[4]+=s*z;z=a[3]*b[1]-a[1]*b[3];q[2]+=k*z;q[4]+=s*z;z=a[4]*b[1]-a[1]*b[4];q[2]+=r*z;q[3]+=r*z;z=a[4]*b[2]-a[2]*b[4];q[3]+=r*z;q[1]+=r*z;z=a[4]*b[3]-a[3]*b[4];q[1]+=r*z;q[2]+=r*z;return q;}
inline Vec static_field(const Vec&p,double g){Vec A{-.5,-2,.5,-1,-.5*g},B{1,0,2,0,-.5*g};return add(product(A,p,g),product(B,product(p,p,g),g));}
template<class F> Vec rk4(double t,const Vec&y,double h,const F&f){Vec k1=f(t,y),k2=f(t+h/2,add(y,scale(k1,h/2))),k3=f(t+h/2,add(y,scale(k2,h/2))),k4=f(t+h,add(y,scale(k3,h)));Vec z{};for(int i=0;i<5;i++)z[i]=y[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;return z;}
struct Protocol{std::string name;double c,a;};
inline void gp(double t,double w,const Protocol&P,double&g,double&gd){g=P.c+P.a*std::sin(w*t);gd=P.a*w*std::cos(w*t);} 
inline Vec hidden_field(double t,const Vec&q,double w,const Protocol&P){double g,gd;gp(t,w,P,g,gd);Vec p{q[0],q[1],q[2],q[3],g*q[4]},f=static_field(p,g);return {f[0],f[1],f[2],f[3],(f[4]-gd*q[4])/g};}
void run_static(const std::string&out,double g,double start,double end,const std::string&tag){double h=.01;Vec p{.5,-.5,-.5,.5,-.5*g};auto f=[g](double,const Vec&x){return static_field(x,g);};long trans=llround(start/h),steps=llround((end-start)/h);double t=0;for(long n=0;n<trans;n++){p=rk4(t,p,h,f);t+=h;if(!ok(p)){std::cerr<<"static blowup\n";std::exit(3);}}
 std::ofstream ts(out+"/static_"+tag+"_timeseries.csv"),pc(out+"/static_"+tag+"_poincare.csv");ts<<"t,w,x,y,z,u,radius\n";pc<<"t,w,y,z,u\n";long rows=0,cross=0;Vec prev=p;double tprev=t;auto wr=[&](){ts<<std::setprecision(17)<<t;for(double v:p)ts<<','<<v;ts<<','<<nv(p)<<"\n";rows++;};wr();
 for(long n=1;n<=steps;n++){prev=p;tprev=t;p=rk4(t,p,h,f);t+=h;if(!ok(p)){std::cerr<<"static blowup record\n";std::exit(4);}if(prev[1]<0&&p[1]>=0){double a=-prev[1]/(p[1]-prev[1]),tc=tprev+a*h;Vec z{};for(int i=0;i<5;i++)z[i]=prev[i]+a*(p[i]-prev[i]);pc<<std::setprecision(17)<<tc<<','<<z[0]<<','<<z[2]<<','<<z[3]<<','<<z[4]<<"\n";cross++;}wr();}
 std::ofstream m(out+"/static_"+tag+"_metadata.txt");m<<std::setprecision(17)<<"gamma="<<g<<"\ndt="<<h<<"\nretained_start="<<start<<"\nretained_end="<<end<<"\ntrajectory_rows="<<rows<<"\nupward_x_crossings="<<cross<<"\n";std::cout<<"static "<<g<<" rows="<<rows<<" crossings="<<cross<<"\n";}
void run_protocol(const std::string&out,const Protocol&P,double w,double transient,double duration,double h,int sample,const std::string&tag){Vec q{.5,-.5,-.5,.5,-.5};auto f=[&](double t,const Vec&x){return hidden_field(t,x,w,P);};long trans=llround(transient/h),steps=llround(duration/h);double t=0;for(long n=0;n<trans;n++){q=rk4(t,q,h,f);t+=h;if(!ok(q)){std::cerr<<"protocol transient blowup "<<tag<<"\n";std::exit(5);}}
 std::ofstream ts(out+"/"+tag+"_timeseries.csv");ts<<"t,w,x,y,z,u_active,gamma,q4_hidden,radius\n";long rows=0;auto wr=[&](){double g,gd;gp(t,w,P,g,gd);Vec p{q[0],q[1],q[2],q[3],g*q[4]};ts<<std::setprecision(17)<<t<<','<<p[0]<<','<<p[1]<<','<<p[2]<<','<<p[3]<<','<<p[4]<<','<<g<<','<<q[4]<<','<<nv(p)<<"\n";rows++;};wr();
 for(long n=1;n<=steps;n++){q=rk4(t,q,h,f);t+=h;if(!ok(q)){std::cerr<<"protocol record blowup "<<tag<<" at "<<t<<"\n";std::exit(6);}if(n%sample==0)wr();}
 std::ofstream m(out+"/"+tag+"_metadata.txt");m<<std::setprecision(17)<<"protocol="<<P.name<<"\ncenter="<<P.c<<"\namplitude="<<P.a<<"\nomega="<<w<<"\nintegration_dt="<<h<<"\noutput_dt="<<h*sample<<"\ntransient="<<transient<<"\nretained_duration="<<duration<<"\ntrajectory_rows="<<rows<<"\n";std::cout<<tag<<" rows="<<rows<<"\n";}
int main(int argc,char**argv){if(argc!=2){std::cerr<<"Usage: generate_dense_riccati OUTPUT_DIRECTORY\n";return 2;}std::string out=argv[1];std::filesystem::create_directories(out);run_static(out,.60,2500,4500,"gamma_0p60");run_static(out,.70,1900,2500,"gamma_0p70");Protocol I{"I",.68,.08},II{"II",.525,.475};run_protocol(out,I,.2,700,500,.005,2,"protocol_I_omega_0p2");run_protocol(out,I,.8,700,500,.005,2,"protocol_I_omega_0p8");run_protocol(out,II,.2,700,500,.005,2,"protocol_II_omega_0p2");run_protocol(out,II,.8,700,500,.005,2,"protocol_II_omega_0p8");run_protocol(out,I,.8,1200,1150,.005,1,"protocol_I_omega_0p8_dense");run_protocol(out,II,.2,1200,1150,.005,1,"protocol_II_omega_0p2_dense");}
