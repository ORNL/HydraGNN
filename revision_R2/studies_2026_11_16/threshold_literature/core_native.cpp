// Small, deterministic constrained GPD likelihood optimizer for bootstrap use.
// Inputs are strictly positive excesses; scale normalization is internal.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
constexpr double lo = -0.49, hi = 0.95;
struct Point { double xi, ls, f; };
double objective(const double* y, int n, double mean, double xi, double ls) {
    if (!std::isfinite(xi) || !std::isfinite(ls) || xi < lo || xi > hi || ls < -25 || ls > 25)
        return 1e100;
    const double s = std::exp(ls);
    double total=0;
    for (int i=0;i<n;++i) {
        const double z=y[i]/mean/s;
        if (1+xi*z <= 0) return 1e100;
        if (std::abs(xi)<1e-7 && std::abs(xi*z)<1e-4) {
            // Taylor expansion continuous through the exponential limit.
            total += z + xi*(z-z*z/2) + xi*xi*(z*z*z/3-z*z/2);
        } else total += (1+1/xi)*std::log1p(xi*z);
    }
    return n*ls+total;
}
Point make(const double* y,int n,double mean,double xi,double ls) {
    xi=std::clamp(xi,lo,hi);
    ls=std::clamp(ls,-25.0,25.0);
    return {xi,ls,objective(y,n,mean,xi,ls)};
}
Point simplex(const double* y,int n,double mean,double ymax,double initial,int& converged,int& iterations) {
    const double scale=std::max(1-initial, std::max(0.0,-initial*ymax/mean)*1.2);
    double step=(initial>0.8 ? -0.12 : 0.12);
    Point p[3]={make(y,n,mean,initial,std::log(scale)),
                make(y,n,mean,initial+step,std::log(scale)),
                make(y,n,mean,initial,std::log(scale)+0.12)};
    converged=0;
    for (iterations=0;iterations<500;++iterations) {
        std::sort(p,p+3,[](const Point& a,const Point& b){return a.f<b.f;});
        const double spread=std::max({std::abs(p[2].xi-p[0].xi),std::abs(p[1].xi-p[0].xi),
                                      std::abs(p[2].ls-p[0].ls),std::abs(p[1].ls-p[0].ls)});
        if (spread<2e-8 && p[2].f-p[0].f < 1e-9*std::max(1.0,std::abs(p[0].f))) {
            converged=1; break;
        }
        const double cx=(p[0].xi+p[1].xi)/2, cs=(p[0].ls+p[1].ls)/2;
        Point r=make(y,n,mean,2*cx-p[2].xi,2*cs-p[2].ls);
        if (r.f<p[0].f) {
            Point e=make(y,n,mean,cx+2*(r.xi-cx),cs+2*(r.ls-cs));
            p[2]=(e.f<r.f?e:r);
        } else if (r.f<p[1].f) p[2]=r;
        else {
            const bool outside=r.f<p[2].f;
            Point c=outside?make(y,n,mean,cx+.5*(r.xi-cx),cs+.5*(r.ls-cs))
                           :make(y,n,mean,cx+.5*(p[2].xi-cx),cs+.5*(p[2].ls-cs));
            if (c.f < (outside?r.f:p[2].f)) p[2]=c;
            else {
                p[1]=make(y,n,mean,p[0].xi+.5*(p[1].xi-p[0].xi),p[0].ls+.5*(p[1].ls-p[0].ls));
                p[2]=make(y,n,mean,p[0].xi+.5*(p[2].xi-p[0].xi),p[0].ls+.5*(p[2].ls-p[0].ls));
            }
        }
    }
    std::sort(p,p+3,[](const Point& a,const Point& b){return a.f<b.f;});
    return p[0];
}
}

extern "C" void fit_batch(const double* x,int batches,int n,double* out) {
    for(int b=0;b<batches;++b) {
        const double* y=x+static_cast<std::size_t>(b)*n;
        double sum=0, squares=0, ymax=0;
        bool valid=n>=2;
        for(int i=0;i<n;++i) {
            valid=valid && std::isfinite(y[i]) && y[i]>0;
            sum+=y[i]; squares+=y[i]*y[i]; ymax=std::max(ymax,y[i]);
        }
        const double mean=sum/n;
        double* z=out+static_cast<std::size_t>(b)*6;
        if(!valid || !(mean>0)) { z[0]=0;z[1]=1;z[2]=1e100;z[3]=0;z[4]=0;z[5]=0;continue; }
        double var=std::max(0.0,squares/n-mean*mean);
        double mom=var>0?0.5*(1-mean*mean/var):lo;
        mom=std::clamp(mom,lo,hi);
        int conv0=0,iter0=0,conv1=0,iter1=0;
        Point a=simplex(y,n,mean,ymax,0.0,conv0,iter0);
        Point c=simplex(y,n,mean,ymax,mom,conv1,iter1);
        Point best=a.f<c.f?a:c;
        const int conv=a.f<c.f?conv0:conv1;
        z[0]=best.xi; z[1]=mean*std::exp(best.ls); z[2]=best.f+n*std::log(mean);
        z[3]=conv && best.f<1e90; z[4]=(best.xi-lo<1e-6 || hi-best.xi<1e-6);z[5]=iter0+iter1;
    }
}
