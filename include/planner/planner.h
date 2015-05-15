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


#ifndef PLANNER_H
#define PLANNER_H

#include "utils/base_classes.h"
#include "utils/math.h"
#include "utils/kalman.h"
#include "data_processing/segmentation.h"

class TangentBug{
public:

    RState                     robot_now;
    RState                     robot_fut;
    SegmentDataExtPtrVectorPtr seg_ext_now;
    SegmentDataExtPtrVectorPtr seg_ext_ftr;
    FrameTf                    n2f;
    FrameTf                    w2r;

    double        d_followed_fin;
    ObjectDataPtr o_followed_fin;
    int         dir_followed_fin;
    KInp          cmd_vel;
    xy            full_potential;
    polar         target;
    polar         target_to_follow;

    double d_to_obj;


    template <class SegData>
    void potential_weight(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_r2n, int frame, xy & full_pot);
    template <class SegData>
    void tangent_bug(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_r2n, int frame, RState rob_pos,
                     double d_followed_old, ObjectDataPtr & o_followed_old, double & d_followed_new, ObjectDataPtr & o_followed_new);
    void run(double _pred_time, KInp u);

    //Constructors & Destructors
    TangentBug(RecfgParam& param, SensorTf& _tf_sns, KalmanSLDM & _k, Segmentation & _segmentation);
    ~TangentBug(){}
private:

    void vel_controller(SegmentDataExtPtrVectorPtr &data, xy & full_pot, polar target, KInp u);
    void find_dir_followed(SegmentDataExtPtrVectorPtr &data, int frame, int   dir_followed_old, ObjectDataPtr o_followed_old,
                                      int & dir_followed    , ObjectDataPtr & o_followed    , double d_followed, polar & target);

    SensorTf   & tf_sns;
    KalmanSLDM & k;
    double     & sensor_range_max;
    double     & angle_max;
    double     & angle_min;
    Segmentation & segmentation;

    double     & pot_scale;
    double     & w_kp_goal;
    double     & w_kd_goal;
    double     & v_kp_w;
    double     & v_kp_goal;
    double     & v_kd_goal;
    double     & v_max;
    double     & w_max;
};

#endif // PLANNER_H
