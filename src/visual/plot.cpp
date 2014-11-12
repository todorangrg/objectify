#include "visual/plot.h"
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>


using namespace cv;


Plot::Plot(std::string window_name):
    green         (   0, 255,   0),
    green_bright  (  51, 255,  51),
    green_dark    (   0, 102,   0),
    red           (   0,   0, 255),
    blue          ( 255,   0,   0),
    blue_bright   ( 255,  51,  51),
    blue_dark     ( 139,   0,   0),
    orange        (   0, 128, 255),
    yellow        (   0, 255, 255),
    cyan          ( 255, 255,   0),
    magenta       ( 255,   0, 255),
    gray          ( 128, 128, 128),
    black         (   0,   0,   0),
    white         ( 255, 255, 255),
    wndView_(window_name){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Plot::putArrow(xy tail,xy tip,Scalar color,int thickness){
    line(plot,tip,tail,color,thickness);
    double angle = atan2((double)tail.y-tip.y, (double)tail.x-tip.x);
    tail.x = (int) ( tip.x +  12.0 * cos(angle + M_PI/4));
    tail.y = (int) ( tip.y +  12.0 * sin(angle + M_PI/4));
    line(plot,tail,tip,color,thickness);
    tail.x = (int) ( tip.x +  12.0 * cos(angle - M_PI/4));
    tail.y = (int) ( tip.y +  12.0 * sin(angle - M_PI/4));
    line(plot,tail,tip,color,thickness);

}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Plot::putFullCircle(xy point,int min_radius,int max_radius,Scalar color){
    for(int i=min_radius;i<=max_radius;i++){
        circle(plot, point,i,color);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

cv::RotatedRect Plot::cov2rect(cv::Matx<double, 2, 2> _C,xy _center) {
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

///------------------------------------------------------------------------------------------------------------------------------------------------///

Point Plot::w2i(double _x, double _y) {
    Mat_<double> pw = (Mat_<double>(3,1) << _x, _y, 1.0);
    Mat_<double> pi = Mw2i * pw;
    return  Point (pi(0,0), pi(1,0));
}
