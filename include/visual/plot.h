#ifndef PLOT_H
#define PLOT_H

#include "utils/base_classes.h"


class ConvolInfo;

class Plot{
public:
    cv::Scalar green;
    cv::Scalar green_bright;
    cv::Scalar green_dark;
    cv::Scalar red;
    cv::Scalar blue;
    cv::Scalar blue_bright;
    cv::Scalar blue_dark;
    cv::Scalar orange;
    cv::Scalar yellow;
    cv::Scalar cyan;
    cv::Scalar magenta;
    cv::Scalar gray;
    cv::Scalar black;
    cv::Scalar white;

    cv::Mat     plot;

    void putArrow         (xy tail, xy tip, cv::Scalar color, int thickness=1);
    void putFullCircle    (xy point, int min_radius, int max_radius, cv::Scalar color);

    cv::RotatedRect cov2rect(cv::Matx<double, 2, 2> _C, xy _center) ;

    virtual void update() = 0;

    Plot(std::string window_name);

    cv::Point w2i(double _x, double _y);
    cv::Point w2i(const xy &_p) { return w2i(_p.x, _p.y); }
protected:
    std::string wndView_;  // opencv window name
    cv::Mat_<double> Mw2i;

};

#endif // PLOT_H
