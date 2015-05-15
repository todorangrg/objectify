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


    seg_oi        ( 176, 160,   0),
    seg_oe        ( 187, 103,  23),
    seg_ni        (  65, 104, 235),
    seg_ne        (  53,  61, 204),
    seg_o2n       ( 141,  67,  71),
    seg_n2o       (  24,  44, 122),
    cov_x         (   2, 199, 248),
    cov_v         (  76, 136,  77),

    obj_gen_0     ( 155, 100,  59),
    obj_gen_1     (  28, 174, 184),
    obj_gen_2     (  37,  78, 255),
    obj_gen_3     (  36, 196, 181),
    obj_gen_4     ( 147,  32, 220),
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

///------------------------------------------------------------------------------------------------------------------------------------------------///

xy Plot::i2w(double _x, double _y) {
    Mat_<double> pw = (Mat_<double>(3,1) << _x, _y, 1.0 / Mi2w(2,2)/*1.0*/);
    Mat_<double> pi = Mi2w * pw;
    return  xy (pi(0,0), pi(1,0));
}
