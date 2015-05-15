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


#include "data_processing/dp.h"
#include "utils/iterators.h"
#include "fstream"

using namespace cv;

DataProcessing::DataProcessing(RecfgParam& param,SensorTf& _tf_sns,PlotData& plot,PlotConv& plot_conv , rosbag::Bag &bag) :
    param(param),
    preprocessing(param,plot),
    k(param, _tf_sns, bag),
    segmentation(param,_tf_sns, plot, k),
    correlation(param,plot,plot_conv),
    planner(param,_tf_sns,k, segmentation),
    tf_sns(_tf_sns),
    plot(plot){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void DataProcessing::run(bool new_frame){
    ros::Time t0;
    std::stringstream info;

    int p_size = 0;
    if(input.sensor_raw){ p_size += input.sensor_raw->size(); }
    info.str(""); info<<"[ n]inp raw  point nr      = "<< p_size            ; plot.putInfoText(info,0,plot.black);
    p_size = 0;
    if(k.seg_init)      { for(int i = 0; i < k.seg_init->size(); i++){ p_size += k.seg_init->at(i)->p.size(); } ; }
    info.str(""); info<<"[ n]mem init point nr      = "<< p_size            ; plot.putInfoText(info,0,plot.black);
    info.str(""); info<<"[ms]Full cycle total time  = "<<input.u.dt * 1000.0; plot.putInfoText(info,0,plot.black);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    t0 = ros::Time::now();
    k.prediction(k.seg_init, input.u);
    info.str(""); info<<"[ms]Kalman pred run time  = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    t0 = ros::Time::now();
    preprocessing.run(input);
    info.str(""); info<<"[ms]Preproc run time      = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    preprocessing.plot_data(input,plot.gray,plot.red,plot.red,plot.green);
    info.str(""); info<<"[ms]Preproc plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    t0 = ros::Time::now();
    segmentation.run(input, k, new_frame);
    info.str(""); info<<"[ms]Segment run time      = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    segmentation .plot_data(input, k, plot.seg_oi, plot.seg_oe, plot.seg_ni, plot.seg_ne);
    info.str(""); info<<"[ms]Segment plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mod
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    t0 = ros::Time::now();
    correlation.run(input, k);
    info.str(""); info<<"[ms]Innov run time        = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    correlation   .plot_all_data(input, k , plot.seg_o2n, plot.seg_n2o);
    info.str(""); info<<"[ms]Innov plot run time   = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    t0 = ros::Time::now();
    k.run(input, correlation.neigh_data_ext[FRAME_OLD], correlation.neigh_data_ext[FRAME_NEW], correlation.neigh_data_init[FRAME_OLD], correlation.neigh_data_init[FRAME_NEW], segmentation);
    info.str(""); info<<"[ms]Kalman run time       = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    plot.plot_kalman(k.seg_init,k, plot.cov_v, plot.cov_x);
    info.str(""); info<<"[ms]Kalman plot run time  = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mod
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     t0 = ros::Time::now();
//     planner.run(0.3, input.u);
//     info.str(""); info<<"[ms]Planner run time      = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    //plot.plot_segm(planner.seg_ext_now, plot.black, false);
    //k.plotw.plot_t_bug(planner.d_followed_fin, planner.o_followed_fin, planner.dir_followed_fin, planner.target, planner.target_to_follow, to_polar(planner.full_potential));

    SensorTf tf_r; tf_r.init(xy(planner.robot_now.xx, planner.robot_now.xy), planner.robot_now.xphi);
    //k.plotw.plot_segm(planner.seg_ext_ftr, k.plotw.black, tf_r);


}
