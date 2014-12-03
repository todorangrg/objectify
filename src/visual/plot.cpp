#include "visual/plot.h"
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>


using namespace cv;


Plot::Plot(std::string window_name):
    green         (   0, 255,   0),
    green_bright  (  51, 255,  51),
    green_dark    (   0, 102,   0),
    red           (   0,   0, 255),
    red_dark      (   0,   0, 102),
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

Point Plot::w2i(double _x, double _y) {
    Mat_<double> pw = (Mat_<double>(3,1) << _x, _y, 1.0);
    Mat_<double> pi = Mw2i * pw;
    return  Point (pi(0,0), pi(1,0));
}
