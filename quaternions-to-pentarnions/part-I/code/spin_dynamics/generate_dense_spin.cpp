#include <filesystem>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

using Vec = std::array<double,3>;
constexpr double PI = 3.141592653589793238462643383279502884;

inline double norm(const Vec& y){return std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);}
inline Vec add(const Vec&a,const Vec&b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline Vec scale(const Vec&a,double s){return {s*a[0],s*a[1],s*a[2]};}
inline Vec cross(const Vec&a,const Vec&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline void normalize(Vec& y){double n=norm(y); for(double &v:y)v/=n;}
inline double gamma_fun(double t,double O){return 0.5*(1.0+std::sin(O*t));}
inline double gamma_dot(double t,double O){return 0.5*O*std::cos(O*t);}
inline void inertia(double g,double&A,double&B){double g2=g*g;A=3.0*(g2*g2+2*g2+3)/((1+g2)*(3+g2));B=(4*g2+3)/(1+g2);}
inline void inertia_derivative(double g,double&dA,double&dB){
  dA=12*g*(std::pow(g,4)-3)/(std::pow(g*g+1,2)*std::pow(g*g+3,2));
  dB=2*g/std::pow(1+g*g,2);
}
inline Vec field(double g){return {1/std::sqrt(3+g*g),1/std::sqrt(2.0),1/std::sqrt(6.0)};}
inline Vec rhs(double t,const Vec&y,double O){double g=gamma_fun(t,O),A,B; inertia(g,A,B); Vec f=field(g); Vec grad={A*y[0]-f[0],B*y[1]-f[1],B*y[2]-f[2]}; return cross(y,grad);}
inline double energy(double t,const Vec&y,double O){double g=gamma_fun(t,O),A,B; inertia(g,A,B); Vec f=field(g); return .5*(A*y[0]*y[0]+B*(y[1]*y[1]+y[2]*y[2]))-(f[0]*y[0]+f[1]*y[1]+f[2]*y[2]);}
inline double power(double t,const Vec&y,double O){double g=gamma_fun(t,O),dA,dB; inertia_derivative(g,dA,dB); double dH=.5*dA*y[0]*y[0]+.5*dB*(y[1]*y[1]+y[2]*y[2])+g*y[0]/std::pow(3+g*g,1.5); return dH*gamma_dot(t,O);}
inline Vec step(double t,const Vec&y,double h,double O,bool project=true){
 Vec k1=rhs(t,y,O); Vec k2=rhs(t+h/2,add(y,scale(k1,h/2)),O); Vec k3=rhs(t+h/2,add(y,scale(k2,h/2)),O); Vec k4=rhs(t+h,add(y,scale(k3,h)),O);
 Vec z={y[0]+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,y[1]+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,y[2]+h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6}; if(project)normalize(z); return z;
}

void run_case(const std::string& out,const std::string& tag,double O,int spp,int transient_periods,int retained_periods,long retained_rows,double display_window=60.0){
 const double T=2*PI/O, h=T/spp; const long trans_steps=1L*transient_periods*spp, rec_steps=1L*retained_periods*spp; const long stride=rec_steps/(retained_rows-1);
 if(stride*(retained_rows-1)!=rec_steps){std::cerr<<"bad stride\n";std::exit(2);} Vec y={.45,.65,-.61}; normalize(y); double t=0;
 for(long n=0;n<trans_steps;n++){y=step(t,y,h,O,true);t+=h;}
 const double t0=t,H0=energy(t,y,O); double W=0, maxerr=0, maxnorm=0; double pprev=power(t,y,O);
 std::ofstream ts(out+"/"+tag+"_timeseries.csv"), pc(out+"/"+tag+"_poincare.csv"), audit(out+"/"+tag+"_work_audit.csv"), display(out+"/"+tag+"_display_window.csv");
 ts<<"t,y_h,y_b,y_c,gamma,H,a,b,c,d,norm_error,work_integral,work_error\n";
 display<<"t,y_h,y_b,y_c,gamma,H,a,b,c,d,norm_error,work_integral,work_error\n";
 pc<<"period,t,y_h,y_b,y_c,gamma\n";
 audit<<"t,H_minus_H0,work_integral,work_error,power\n";
 auto row_string=[&](){double g=gamma_fun(t,O),den=std::sqrt(3+g*g),H=energy(t,y,O),ne=std::abs(norm(y)-1),err=(H-H0)-W; maxnorm=std::max(maxnorm,ne);maxerr=std::max(maxerr,std::abs(err));
   std::ostringstream os; os<<std::setprecision(17)<<t<<','<<y[0]<<','<<y[1]<<','<<y[2]<<','<<g<<','<<H<<','<<y[0]*std::sqrt(3.0)/den<<','<<y[1]<<','<<y[2]<<','<<y[0]*g/den<<','<<ne<<','<<W<<','<<err; return os.str();};
 auto write_row=[&](){auto row=row_string(); ts<<row<<"\n"; double H=energy(t,y,O),err=(H-H0)-W; audit<<std::setprecision(17)<<t<<','<<(H-H0)<<','<<W<<','<<err<<','<<power(t,y,O)<<"\n";};
 write_row(); long rows=1, display_rows=0;
 const long display_steps=std::min(rec_steps, static_cast<long>(std::ceil(display_window/h)));
 const long display_start=rec_steps-display_steps;
 for(long n=1;n<=rec_steps;n++){
   Vec yn=step(t,y,h,O,true); double tn=t+h; double pnow=power(tn,yn,O); W += .5*h*(pprev+pnow); y=yn;t=tn;pprev=pnow;
   if(n%stride==0){write_row();rows++;}
   if(n>=display_start){display<<row_string()<<"\n"; display_rows++;}
   if(n%spp==0){long per=n/spp; pc<<std::setprecision(17)<<per<<','<<t<<','<<y[0]<<','<<y[1]<<','<<y[2]<<','<<gamma_fun(t,O)<<"\n";}
 }
 std::ofstream meta(out+"/"+tag+"_metadata.txt"); meta<<std::setprecision(17)
  <<"omega="<<O<<"\nforcing_period="<<T<<"\nsteps_per_period="<<spp<<"\ndt="<<h<<"\ntransient_periods="<<transient_periods
  <<"\nretained_periods="<<retained_periods<<"\ntrajectory_rows="<<rows<<"\nstroboscopic_rows="<<retained_periods
  <<"\ndisplay_window="<<display_window<<"\ndisplay_rows="<<display_rows
  <<"\nmax_norm_error="<<maxnorm<<"\nmax_work_identity_error="<<maxerr<<"\n";
 std::cout<<tag<<": rows="<<rows<<" display="<<display_rows<<" poincare="<<retained_periods<<" workerr="<<std::setprecision(12)<<maxerr<<"\n";
}
int main(int argc, char* argv[]){
 if(argc>2){std::cerr<<"Usage: generate_dense_spin [output_directory]\n"; return 2;}
 const std::string out=argc==2 ? argv[1] : "outputs/spin_dense";
 std::error_code ec; std::filesystem::create_directories(out,ec);
 if(ec){std::cerr<<"Cannot create output directory: "<<ec.message()<<"\n"; return 1;}
 {std::ofstream probe(out+"/.write_test"); if(!probe){std::cerr<<"Output directory is not writable\n"; return 1;}}
 std::filesystem::remove(out+"/.write_test",ec);
 run_case(out,"regular",.30,800,300,900,24001);
 run_case(out,"chaotic",1.20,910,500,3000,91001);
 return 0;
}
