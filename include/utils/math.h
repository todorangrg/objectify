/***************************************************************************
 *   Software License Agreement (BSD License)                              *
 *   Copyright (C) 2015 by Horatiu George Todoran <todorangrg@gmail.com>   *
 *                                                                         *
 *   Redistribution and use in source and binary forms, with or without    *
 *   modification, are permitted provided that the following conditions    *
 *   are met:                                                              *
 *                                                                         *
 *   1. Redistributions of source code must retain the above copyright     *
 *      notice, this list of conditions and the following disclaimer.      *
 *   2. Redistributions in binary form must reproduce the above copyright  *
 *      notice, this list of conditions and the following disclaimer in    *
 *      the documentation and/or other materials provided with the         *
 *      distribution.                                                      *
 *   3. Neither the name of the copyright holder nor the names of its      *
 *      contributors may be used to endorse or promote products derived    *
 *      from this software without specific prior written permission.      *
 *                                                                         *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   *
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     *
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS     *
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE        *
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,  *
 *   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
 *   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      *
 *   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      *
 *   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT    *
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY *
 *   WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           *
 *   POSSIBILITY OF SUCH DAMAGE.                                           *
 ***************************************************************************/


#ifndef OBJ_MATH_H
#define OBJ_MATH_H


#include <opencv/cv.h>
#include <math.h>

class PointData;
class PointDataCpy;
typedef cv::Point2d xy;

///------------------------------------------------------------------------------------------------------------------------------------------------///

class Distributions {
public:

    static double normalDist();
    static double normalDist(double sigma);
    static double normalDist(double mean, double sigma);
    static double normalDistAbs(double mean, double sigma);
    static void normalDist(cv::Vec<double,3> &mean, cv::Vec<double,3> sigma);

    static double uniformDist(double min, double max);
    static void uniformDist(double min, double max, cv::Vec<double,3> &des);
};

class Gauss{
public:

    void   add_w_sample(double val, double w);
    double getMean();
    double getVariance();
    double getSampleVariance();

    Gauss():entry_no(0),sumweight(0),mean(0),m2(0){}
    Gauss(double _entry_no,double _sumweight, double _mean, double _m2):entry_no(_entry_no),sumweight(_sumweight),mean(_mean),m2(_m2){}
private:

    int    entry_no;
    double sumweight;
    double mean;
    double m2;
};

class NormalFunc{
public:

    void set(double _mean, double _deviation, double _no_sigma_dev);
    double f(double x);
private:

    double mean;
    double sigma;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class Line{
public:

    double a;
    double b;
    double c;
    Line(double aa,double bb, double cc){
        a=aa;b=bb;c=cc;
    }
    Line(){}
};

Line get_line_param (xy p1, xy p2);

xy   get_line_inters(Line l,xy p);

xy   get_line_inters(Line l1, Line l2);

double get_dist_p(Line l, xy p0, xy *p_d);

///------------------------------------------------------------------------------------------------------------------------------------------------///

class polar{
public:

    double r;
    double angle;
    polar();
    polar(double _r, double _angle);
    polar(const polar &_p);
    polar(PointData &p);
};

xy    to_xy(polar p      );
xy    to_xy(PointData &p);

polar to_polar(xy c);

polar polar_diff(polar p1     , polar p2     );
polar polar_diff(polar p1     , PointData &p2);
polar polar_diff(PointData &p1, polar p2     );
polar polar_diff(PointData p1 , PointData &p2);

polar radial_diff(polar p1, polar p2     );//p1.r==p2.r
polar radial_diff(polar p1, PointData &p2);



///------------------------------------------------------------------------------------------------------------------------------------------------///

double sqr(double x);
double normalizeAngle(double &angle);
double deg_to_rad(double angle_deg);
double rad_to_deg(double angle_rad);
double sgn(double x);

inline double diff(polar p1, polar p2){return sqrt(sqr(p1.r) + sqr(p2.r) - 2.0 * p1.r * p2.r *cos(p1.angle - p2.angle));}
inline double diff(xy p1   , xy p2   ){return sqrt(sqr(p1.x - p2.x) + sqr(p1.y - p2.y));}


void angular_bounds(polar p        , double circle_rad, double* search_angle);
void angular_bounds(PointData &p   , double circle_rad, double* search_angle);
void angular_bounds(PointDataCpy &p, double circle_rad, double* search_angle);

///------------------------------------------------------------------------------------------------------------------------------------------------///

void set_tf_mat(cv::Matx<double,3,3>& tf, xy trans, double rot);
xy   mat_mult  (cv::Matx<double,3,3>& tf, xy p);

cv::RotatedRect cov2rect(cv::Matx<double, 2, 2> _C,xy _center);

#endif // OBJ_MATH_H
