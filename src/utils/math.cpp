#include "utils/base_classes.h"
#include "utils/math.h"
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>


using namespace cv;

boost::minstd_rand gIntGenStd;
boost::mt19937 gIntGen19937;
boost::variate_generator<boost::mt19937, boost::normal_distribution<> > gNormal(gIntGen19937, boost::normal_distribution<>(0,1));
boost::uniform_01<boost::minstd_rand> gUniform(gIntGenStd);

double  Distributions::normalDist() {
    return gNormal();
}
double  Distributions::normalDist(double sigma) {
    return sigma * gNormal();
}
double  Distributions::normalDist(double mean, double sigma) {
    return mean + sigma * gNormal();
}
void Distributions::normalDist(cv::Vec<double,3> &mean, cv::Vec<double,3> sigma) {
    mean[0] = normalDist(mean[0], sigma[0]);
    mean[1] = normalDist(mean[1], sigma[1]);
    mean[2] = normalDist(mean[2], sigma[2]);
}
double Distributions::uniformDist(double min, double max){
    double s = max - min;
    return min + s * gUniform();
}
void Distributions::uniformDist(double min, double max, cv::Vec<double,3> &des){
    des[0] = uniformDist(min, max);
    des[1] = uniformDist(min, max);
    des[2] = uniformDist(min, max);
}


void Gauss::add_w_sample(double val, double w){
    entry_no++;
    double temp  = w + sumweight;
    double delta = val - mean;
    double r = delta * w /temp;
    mean += r;
    m2 += sumweight * delta * r;
    sumweight = temp;
}

double Gauss::getMean(){
    return mean;
}
double Gauss::getVariance(){
    if(entry_no<=1){
        //std::cout<<"WARNING: requesting variance of less than 2 entries"<<std::endl;
        return 0;
    }
    return (m2/sumweight)* (double)entry_no/((double)entry_no - 1.0);
}
double Gauss::getSampleVariance(){
    return m2/sumweight;
}

void NormalFunc::set(double _mean, double _deviation, double _no_sigma_dev){
    mean = _mean;
    sigma = _deviation / _no_sigma_dev;
}

double NormalFunc::f(double x){
    return std::exp( - sqr( x - mean ) / ( 2.0 * sqr( sigma ) ) );
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

Line get_line_param(xy p1,xy p2){
    double a=p2.y-p1.y;
    double b=p1.x-p2.x;
    double c=-(a*p2.x+b*p2.y);

    return Line(a,b,c);
}
xy get_line_inters(Line l,xy p){
    return xy( (l.b*(l.b*p.x-l.a*p.y)-l.a*l.c)/(sqr(l.a)+sqr(l.b)), (l.a*(-l.b*p.x+l.a*p.y)-l.b*l.c)/(sqr(l.a)+sqr(l.b)));
}

xy get_line_inters(Line l1,Line l2){
    double x=10000,y=10000;
    if(l1.a*l2.b!=l1.b*l2.a){
        x=(l2.b*l1.c-l2.c*l1.b)/(l2.a*l1.b-l1.a*l2.b);
        y=(l2.c*l1.a-l1.c*l2.a)/(l2.a*l1.b-l1.a*l2.b);
        return xy(x,y);
    }
    else if((l1.a==l2.a)&&(l1.b==l2.b)&&(l1.c==l2.c)){
        std::cout<<"paralel overlapping lines when computing get_line_intersection"<<std::endl;
    }
    else{
        std::cout<<"paralel NONoverlapping lines when computing get_line_intersection"<<std::endl;
    }
    return xy(0,0);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///


polar::polar() {
    r=0;angle=0;
}
polar::polar(double _r, double _angle) {
    r=_r;angle=_angle;
}
polar::polar(const polar &_p){
    r=_p.r;angle=_p.angle;
}
polar::polar(PointData &_p){
    r=_p.r;angle=_p.angle;
}



polar polar_diff(polar p1,polar p2){
    return polar(sqrt(sqr(p2.r*cos(p2.angle)-p1.r*cos(p1.angle))+sqr(p2.r*sin(p2.angle)-p1.r*sin(p1.angle))),
                 atan2(p2.r*sin(p2.angle)-p1.r*sin(p1.angle),p2.r*cos(p2.angle)-p1.r*cos(p1.angle)));
}

polar polar_diff(polar p1,PointData &p2){
    return polar_diff(p1,polar(p2.r,p2.angle));
}
polar polar_diff(PointData &p1,polar p2){
    return polar_diff(polar(p1.r,p1.angle),p2);
}

polar polar_diff(PointData p1,PointData &p2){
    return polar_diff(polar(p1.r,p1.angle),polar(p2.r,p2.angle));
}

polar radial_diff(polar p1,polar p2){
    double d_angle=fabs(p1.angle-p2.angle);
    normalizeAngle(d_angle);
    return polar(d_angle*p1.r,d_angle);
}
polar radial_diff(polar p1,PointData &p2){
    return radial_diff(p1,polar(p2.r,p2.angle));
}

xy to_xy(polar p){
    return xy(p.r*cos(p.angle),p.r*sin(p.angle));
}
xy to_xy(PointData &p){
    return to_xy(polar(p.r,p.angle));
}

polar to_polar(xy c){
    return polar(sqrt(c.x*c.x+c.y*c.y),atan2(c.y,c.x));
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

double normalizeAngle(double &angle) {
    while(angle > M_PI){
      angle = angle - (2*M_PI);
    }
    while(angle <= -M_PI){
      angle = angle + (2*M_PI);
    }
    return angle;
}

double deg_to_rad(double angle_deg){
    return (angle_deg*M_PI)/180.0;
}

double rad_to_deg(double angle_rad){
    return (angle_rad*180.0)/(double)M_PI;
}

double sqr(double x){
    return x*x;
}

double sgn(double x){
    if(x>=0){
        return 1;
    }
    else{
        return -1;
    }
}

void angular_bounds(polar p, double circle_rad, double* search_angle){
    search_angle[0] = p.angle;
    search_angle[1] = search_angle[0];
    if( fabs( circle_rad / p.r ) > 1.0  ){
        search_angle[0] -= 2 * M_PI;
        search_angle[1] += 2 * M_PI;
    }
    else{
        search_angle[0] -= asin( circle_rad / p.r );
        search_angle[1] += asin( circle_rad / p.r );
    }
}
void angular_bounds(PointData &p,double circle_rad, double* search_angle){
    return angular_bounds(polar(p.r,p.angle),circle_rad,search_angle);
}

void angular_bounds(PointDataCpy &p,double circle_rad, double* search_angle){
    return angular_bounds(polar(p.r,p.angle),circle_rad,search_angle);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///


void set_tf_mat(cv::Matx<double,3,3>& tf, xy trans, double rot){
    cv::Matx<double,3,3> R ( cos(rot), -sin(rot),   0,
                             sin(rot),  cos(rot),   0,
                                      0,           0,   1);    // Rotate
    cv::Matx<double,3,3> T (          1,           0, trans.x,
                                      0,           1, trans.y,
                                      0,           0,   1);    // Translate
    // Calculate final transformation matrix
    tf = T * R;
    //Mptf = Mptf.inv();
    //std::cout <<  "Ms2r = " << std::endl << Ms2r << std::endl;
    //std::cout <<  "Mr2s = " << std::endl << Mr2s << std::endl;
}

xy   mat_mult(cv::Matx<double,3,3>& tf,xy p){
    cv::Matx<double,3,1> pw (p.x, p.y, 1.0);
    cv::Matx<double,3,1> pi = tf * pw;
    return  xy (pi(0,0), pi(1,0));
}

cv::RotatedRect cov2rect(cv::Matx<double, 2, 2> _C,xy _center){
    cv::RotatedRect ellipse;
    cv::Mat_<double> eigval, eigvec;
    cv::eigen(_C, eigval, eigvec);

    /// Exercise4
    bool index_x;
    ellipse.center = _center;
    if(_C(0,0)>_C(1,1)){
        index_x=0;
    }
    else{
        index_x=1;
    }
    ellipse.size.height=sqrt(fabs(eigval(0,!index_x)))*2.4477;//y
    ellipse.size.width=sqrt(fabs(eigval(0,index_x)))*2.4477;//x
    if((eigval(0,index_x)!=0)&&(eigval(0,!index_x)!=0))
        ellipse.angle=atan2(eigvec(index_x,1),eigvec(index_x,0))*(180/M_PI);

    return ellipse;
}
