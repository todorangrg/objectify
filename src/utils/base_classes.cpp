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


#include "utils/base_classes.h"

#include "utils/kalman.h"


void RecfgParam::init_normal_smooth_mask(){
    smooth_mask.clear();
    smooth_mask.push_back(0);
    for( int i = 1; i <= convol_normals_smooth_mask_size; i++){
        long double val = std::exp( (long double)(- sqr( convol_normals_smooth_mask_size * i * convol_normals_smooth_mask_dist )) ) ;
        smooth_mask.push_back(val);
        smooth_mask.push_front(val);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

RState::RState(cv::Mat _S):
    xx(_S.at<double>(0)), xy(_S.at<double>(1)), xphi(_S.at<double>(2)),
    vlin(_S.at<double>(3)), vang(_S.at<double>(4)),
    alin(_S.at<double>(5)), aang(_S.at<double>(6)){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

OiState::OiState(cv::Mat _S_O):
    xx(_S_O.at<double>(0)), xy(_S_O.at<double>(1)), xphi(_S_O.at<double>(2)),
    vx(_S_O.at<double>(3)), vy(_S_O.at<double>(4)), vphi(_S_O.at<double>(5)),
    ax(_S_O.at<double>(6)), ay(_S_O.at<double>(7)), aphi(_S_O.at<double>(8)){}

void OiState::init(cv::Mat _S_O){
    xx = _S_O.at<double>(0); xy = _S_O.at<double>(1); xphi = _S_O.at<double>(2);
    vx = _S_O.at<double>(3); vy = _S_O.at<double>(4); vphi = _S_O.at<double>(5);
    ax = _S_O.at<double>(6); ay = _S_O.at<double>(7); aphi = _S_O.at<double>(8);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

///TODO DISCARD AND USE TF LIBRARY
void SensorTf::init(xy trans, double rot) {
    cv::Mat_<double> R = (cv::Mat_<double>(3,3) <<  cos(rot),  - sin(rot), 0,
                                                    sin(rot),    cos(rot), 0,
                                                    0,                  0, 1);    // Rotate
    cv::Mat_<double> T = (cv::Mat_<double>(3,3) <<  1, 0, trans.x,
                                                    0, 1, trans.y,
                                                    0, 0, 1);    // Translate

    // Calculate final transformation matrix
    Ms2r = T * R;
    Mr2s = Ms2r.inv();
    //std::cout <<  "Ms2r = " << std::endl << Ms2r << std::endl;
    //std::cout <<  "Mr2s = " << std::endl << Mr2s << std::endl;
}
xy SensorTf::s2r(double _x, double _y) {
    cv::Mat_<double> pw = (cv::Mat_<double>(3,1) << _x, _y, 1.0);
    cv::Mat_<double> pi = Ms2r * pw;
    return  xy(pi(0,0), pi(1,0));
}
xy SensorTf::r2s(double _x, double _y) {
    cv::Mat_<double> pw = (cv::Mat_<double>(3,1) << _x, _y, 1.0);
    cv::Mat_<double> pi = Mr2s * pw;
    return  xy(pi(0,0), pi(1,0));
}

///TODO DISCARD AND USE TF LIBRARY
void FrameTf::init(RState rob_x, RState rob_bar_x){
    double c1 = cos( - rob_bar_x.xphi), s1 = sin( - rob_bar_x.xphi);
    double c2 = cos(   rob_x.xphi),     s2 = sin(   rob_x.xphi);
    double xx = rob_x.xx - rob_bar_x.xx;
    double yy = rob_x.xy - rob_bar_x.xy;
    cv::Mat_<double> R1 = (cv::Mat_<double>(3,3) <<c1, -s1,  0, s1, c1,  0, 0, 0, 1);
    cv::Mat_<double> T  = (cv::Mat_<double>(3,3) << 1,   0, xx,  0,  1, yy, 0, 0, 1);
    cv::Mat_<double> R2 = (cv::Mat_<double>(3,3) <<c2, -s2,  0, s2, c2,  0, 0, 0, 1);
    Mro2rn = R1 * T * R2;
    Mrn2ro = Mro2rn.inv();
    //std::cout <<  "Mrn2ro = " << std::endl << Mrn2ro << std::endl;
}
xy FrameTf::rn2ro(double _x, double _y) {
    cv::Mat_<double> pw = (cv::Mat_<double>(3,1) << _x, _y, 1.0);
    cv::Mat_<double> pi = Mrn2ro * pw;
    return  xy(pi(0,0), pi(1,0));
}
xy FrameTf::ro2rn(double _x, double _y) {
    cv::Mat_<double> pw = (cv::Mat_<double>(3,1) << _x, _y, 1.0);
    cv::Mat_<double> pi = Mro2rn * pw;
    return  xy(pi(0,0), pi(1,0));
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

SegmentDataExt& SegmentDataExt::operator= (const SegmentDataExt& seg){
  return *this;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

CorrInput::CorrInput(SegmentDataExtPtr _frame_old, SegmentDataExtPtr _frame_new, double _stitch_perc, bool _reverse) :
    frame_old(_frame_old),
    frame_new(_frame_new),
    stitch_perc(_stitch_perc),
    reverse(_reverse){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

InputData::InputData(PointDataVectorPtr &input, RState _rob_x, KInp _u, ros::Time _time_stamp) :
    rob_x(_rob_x),
    u(_u),
    time_stamp(_time_stamp),
    sensor_raw(input),
    seg_init(new SegmentDataPtrVector   ),
    seg_ext (new SegmentDataExtPtrVector){
        if(sensor_raw->size() == 0){
            is_valid = false;
        } else{
            is_valid = true;
        }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
