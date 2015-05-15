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


#include "data_processing/dp_node.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include "data_processing/dp.h"
#include "utils/math.h"
#include "visual/plot_data.h"

#include <rosbag/bag.h>

int main( int argc , char **argv ) {

    //debug
    rosbag::Bag bag;


    ros::init( argc , argv , "obj" );
    ros::NodeHandle n;
    RecfgParam dyn_param;
    SensorTf sensor_tf;
    PlotData  plot_data("Object_tracking_visualization",dyn_param, sensor_tf);
    PlotConv  plot_conv("Convolution_visualization",dyn_param);
    DataProcessingNode dp(n, dyn_param, sensor_tf, plot_data,plot_conv, bag);
    dp.frame2frame_callback = false;
    ros::Time t0 = ros::Time::now();

    while ( ros::ok() ) {
        ros::Rate rate( dp.sleep_freq );
        t0 = ros::Time::now();

        if(dp.new_data){

            dp.input = dp.callback_odom_laser_data;
        }
        dp.k.advance(dp.input,dp.new_data);
        dp.run(dp.new_data);
        dp.k.time_stamp = dp.input.time_stamp;
        dp.new_data = false;

        plot_data.update();
        plot_conv.update();
        dp.k.plotw.update();

        std::stringstream info; info.str(""); info<<"[ms]Full cycle busy time  = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot_data.putInfoText(info,0,plot_data.black);//does not freeze value in step sim mode

        bool update = ( dp.frame2frame && dp.frame2frame_callback ) || ( !dp.frame2frame );
        if(update){

            geometry_msgs::Twist cmd;


            cmd.linear.x = dp.planner.cmd_vel.v; cmd.linear.y = 0.; cmd.angular.z = dp.planner.cmd_vel.w; /// creates motion command
//            dp.pub_cmd_.publish(cmd);                                                                     /// publishes motion command

            dp.frame2frame_callback = false; dp.new_data = true;
            dp.unpause_gazebo.call(dp.empty_srv);
            rate.sleep();
            if(dp.frame2frame){ dp.pause_gazebo.call(dp.empty_srv); }
        }
        else{
            dp.pause_gazebo.call(dp.empty_srv);
        }
        ros::spinOnce();
    }
    return 0;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

DataProcessingNode::DataProcessingNode(ros::NodeHandle & n, RecfgParam& param, SensorTf& sns_tf, PlotData& plot, PlotConv& plot_conv , rosbag::Bag &bag):
        DataProcessing( param, sns_tf, plot,plot_conv, bag ), n_( n ),
        odom_sub ( n , "/r1/odom"            , 10 ),/*base_pose_ground_truth*//*odom*/
        laser_sub( n , "/r1/front_laser/scan", 10 ),/*base_scan*/

        sleep_freq(10),
        sync( MySyncPolicy( 10 ) , odom_sub , laser_sub ),
        new_data(false){

    sync.registerCallback( boost::bind( &DataProcessingNode::callback_odom_laser , this , _1 , _2 ) );
    reconfigureFnc_ =      boost::bind( &DataProcessingNode::callbackParameters  , this , _1 , _2 );
    reconfigureServer_.setCallback( reconfigureFnc_ );

    pause_gazebo  =  n.serviceClient<std_srvs::Empty>("gazebo/pause_physics");
    unpause_gazebo = n.serviceClient<std_srvs::Empty>("gazebo/unpause_physics");

    pub_cmd_ = n.advertise<geometry_msgs::Twist>("r1/cmd_vel", 1);
    //unpause_gazebo.call(empty_srv);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

// void DataProcessingNode::callback_odom_laser(const geometry_msgs::TwistConstPtr &_odom , const sensor_msgs::LaserScanConstPtr &_laser ){
//     //new_data= true;  
//     RState stateOdom(0, 0, 0);
//     if(!k.pos_init){ k.init(stateOdom); }
// 
// //    std::cout<<_odom->header.stamp.toSec() - _laser->header.stamp.toSec()<<std::endl; TODO
// 
//     PointDataVectorPtr laser_raw_now( new PointDataVector );
// 
//     int nr = (_laser->angle_max - _laser->angle_min) / _laser->angle_increment;
//     for (int i = 0 ; i < nr ; i++ ) {
//         double laser_noise=nd.normalDist(param.sensor_noise_sigma);
//         double laser_i;
//         if((_laser->ranges[i] < param.sensor_r_max )&&(_laser->ranges[i]>0.18)){
//             laser_i = _laser->ranges[i] + laser_noise;
//             laser_raw_now->push_back( polar( laser_i, _laser->angle_min + ( _laser->angle_increment * i ) ) );
//         }
//     }
//     KInp vel(_odom->linear.x,_odom->angular.z);
// 
//     Distributions noise;
//     if(fabs(vel.v) < 0.01){ vel.v = 0.; }
//     if(fabs(vel.w) < 0.01){ vel.w = 0.; }
//     vel.v += noise.normalDist(0, sim_rob_alfa_1 * vel.v + sim_rob_alfa_2 * vel.w);
//     vel.w += noise.normalDist(0, sim_rob_alfa_3 * vel.v + sim_rob_alfa_4 * vel.w);//noising the input
// 
// //    vel.v = 0;
// //    vel.w = 0;
// 
//     callback_odom_laser_data = InputData(laser_raw_now, stateOdom, vel, _laser->header.stamp);
//     param.cb_sensor_point_angl_inc=_laser->angle_increment;
//     param.cb_sensor_point_angl_max=_laser->angle_max;
//     param.cb_sensor_point_angl_min=_laser->angle_min;
// 
//     k.rob_real = stateOdom;
// }


void DataProcessingNode::callback_odom_laser(const nav_msgs::OdometryConstPtr &_odom , const sensor_msgs::LaserScanConstPtr &_laser ){
    //new_data= true;  
    RState stateOdom(_odom->pose.pose.position.x, _odom->pose.pose.position.y, quaternion2Angle2D(_odom->pose.pose.orientation));
    if(!k.pos_init){ k.init(stateOdom); }

//    std::cout<<_odom->header.stamp.toSec() - _laser->header.stamp.toSec()<<std::endl; TODO

    PointDataVectorPtr laser_raw_now( new PointDataVector );

    int nr = (_laser->angle_max - _laser->angle_min) / _laser->angle_increment;
    for (int i = 0 ; i < nr ; i++ ) {
        double laser_noise=nd.normalDist(param.sensor_noise_sigma);
        double laser_i;
        if((_laser->ranges[i] < param.sensor_r_max )&&(_laser->ranges[i]>0.18)){
            laser_i = _laser->ranges[i] + laser_noise;
            laser_raw_now->push_back( polar( laser_i, _laser->angle_min + ( _laser->angle_increment * i ) ) );
        }
    }
    KInp vel(_odom->twist.twist.linear.x,_odom->twist.twist.angular.z);

    Distributions noise;
    if(fabs(vel.v) < 0.01){ vel.v = 0.; }
    if(fabs(vel.w) < 0.01){ vel.w = 0.; }
    vel.v += noise.normalDist(0, sim_rob_alfa_1 * vel.v + sim_rob_alfa_2 * vel.w);
    vel.w += noise.normalDist(0, sim_rob_alfa_3 * vel.v + sim_rob_alfa_4 * vel.w);//noising the input

//    vel.v = 0;
//    vel.w = 0;

    callback_odom_laser_data = InputData(laser_raw_now, stateOdom, vel, _laser->header.stamp);
    param.cb_sensor_point_angl_inc=_laser->angle_increment;
    param.cb_sensor_point_angl_max=_laser->angle_max;
    param.cb_sensor_point_angl_min=_laser->angle_min;

    k.rob_real = stateOdom;
}

/** converts a quaternion to a rotation matrix : http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
 * Only valid on a 2D xy plane
 * opposite =  2 * z * w
 * adjacent = 1 - 2 * z^2
 * @param quat
 * @return angle on xy plane
 **/
double DataProcessingNode::quaternion2Angle2D ( const geometry_msgs::Quaternion &quat ) {
    double opposite =     2 * quat.z * quat.w;
    double adjacent = 1 - 2 * quat.z * quat.z;

    if      ( opposite >= 0 && adjacent >= 0 ) { return (            asin( opposite ) ); }
    else if ( opposite >= 0 && adjacent <= 0 ) { return ( M_PI     - asin( opposite ) ); }
    else if ( opposite <= 0 && adjacent <= 0 ) { return ( M_PI     - asin( opposite ) ); }
    else                                       { return ( 2 * M_PI + asin( opposite ) ); }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void DataProcessingNode::callbackParameters (objectify::objectify_paramConfig &config , uint32_t level ) {
    sleep_freq = config.sleep_freq;

    static bool frame2frame_switch_old = false;

    sim_rob_alfa_1                        = config.sim_rob_alfa_1;
    sim_rob_alfa_2                        = config.sim_rob_alfa_2;
    sim_rob_alfa_3                        = config.sim_rob_alfa_3;
    sim_rob_alfa_4                        = config.sim_rob_alfa_4;
    param.sensor_r_max                    = config.sim_sensor_r_max;
    param.sensor_noise_sigma              = config.sim_sensor_noise_sigma;

    param.viz_data                        = config.viz_data;
    param.viz_data_grid                   = config.viz_data_grid;
    param.viz_data_raw                    = config.viz_data_raw;
    param.viz_data_preproc                = config.viz_data_preproc;
    param.viz_data_oult_preproc           = config.viz_data_oult_preproc;
    param.viz_data_segm_init              = config.viz_data_segm_init;
    param.viz_data_segm_ext               = config.viz_data_segm_ext ;
    param.viz_data_corr_links             = config.viz_data_corr_links;

    param.viz_convol                      = config.viz_convol;
    param.viz_correl_queue_no             = config.viz_correl_queue_no;
    param.viz_convol_step_no              = config.viz_convol_step_no;
    param.viz_data_tf                     = config.viz_data_tf;
    param.viz_convol_all                  = config.viz_convol_all;
    param.viz_convol_normals              = config.viz_convol_normals;
    param.viz_convol_tf                   = config.viz_convol_tf;
    param.viz_convol_tf_ref2spl           = config.viz_convol_tf_ref2spl;

    param.viz_world                       = config.viz_world;
    param.viz_world_grid                  = config.viz_world_grid;
    param.viz_world_len                   = config.viz_world_len;

    param.preproc_filter                  = config.preproc_filter;
    param.preproc_filter_circle_rad       = config.preproc_filter_circle_rad;
    param.preproc_filter_circle_rad_scale = config.preproc_filter_circle_rad_scale;
    param.preproc_filter_sigma            = config.preproc_filter_sigma;

    param.segm_discont_dist               = config.segm_discont_dist;
    param.segm_outl_circle_rad            = config.segm_outl_circle_rad;
    param.segm_outl_prob_thres            = config.segm_outl_prob_thres / 100.0;
    param.segm_outl_sigma                 = config.segm_outl_sigma;

    param.corr_queue_d_thres              = config.corr_queue_d_thres;
    param.corr_neigh_circle_rad           = config.corr_neigh_circle_rad;

    param.convol_full_search              = config.convol_full_search;
    param.convol_SVD                      = config.convol_SVD;
    param.convol_sample_dist              = config.convol_sample_dist;
    param.convol_min_len_perc             = config.convol_min_len_perc;
    param.convol_marg_extr_excl           = config.convol_marg_extr_excl / 100.0;
    param.convol_normals_smooth_mask_size = config.convol_normals_smooth_mask_size;
    param.convol_normals_smooth_mask_dist = config.convol_normals_smooth_mask_dist;
    param.init_normal_smooth_mask();
    param.convol_key_d_angle              = deg_to_rad(config.convol_key_d_angle);
    param.convol_com_dr_thres             = config.convol_com_dr_thres;
    param.convol_ang_mean_thres           = deg_to_rad(config.convol_ang_mean_thres);//in deg
    param.convol_ang_var_thres            = config.convol_ang_var_thres;
    param.convol_sqr_err_thres            = config.convol_sqr_err_thres;
    param.convol_p_no_perc_thres          = config.convol_p_no_perc_thres / 100.0;
    param.convol_noise_ang_base           = config.convol_noise_ang_base;


    param.kalman_alfa_ini_obj_pow_dt      = config.kalman_alfa_ini_obj_pow_dt;
    param.kalman_alfa_pre_obj_xy_min      = config.kalman_alfa_pre_obj_xy_min;
    param.kalman_alfa_pre_obj_phi         = config.kalman_alfa_pre_obj_phi;
    param.kalman_alfa_dsc_obj_surface     = config.kalman_alfa_dsc_obj_surface;
    param.kalman_discard_old_seg_perc     = config.kalman_discard_old_seg_perc;
    param.kalman_no_upd_vel_hard0         = config.kalman_no_upd_vel_hard0;

    param.kalman_alfa_pre_rob_v_base      = config.kalman_alfa_pre_rob_v_base;
    param.kalman_alfa_pre_rob_w_base      = config.kalman_alfa_pre_rob_w_base;
    param.kalman_alfa_upd_rob_vv          = config.kalman_alfa_upd_rob_vv;
    param.kalman_alfa_upd_rob_ww          = config.kalman_alfa_upd_rob_ww;

    param.kalman_adaptive_resid_min       = config.kalman_adaptive_resid_min;
    param.kalman_adaptive_scale_bound     = config.kalman_adaptive_scale_bound;
    param.kalman_adaptive_noise_scale     = config.kalman_adaptive_noise_scale;

    param.kalman_adpt_obj_resid_scale     = config.kalman_adpt_obj_resid_scale;

    param.kalman_dynamic_obj              = config.kalman_dynamic_obj;

    param.planner_pot_scale               = config.planner_pot_scale;
    param.planner_w_kp_goal               = config.planner_w_kp_goal;
    param.planner_w_kd_goal               = config.planner_w_kd_goal;
    param.planner_v_kp_w                  = config.planner_v_kp_w;
    param.planner_v_kp_goal               = config.planner_v_kp_goal;
    param.planner_v_kd_goal               = config.planner_v_kd_goal;
    param.planner_v_max                   = config.planner_v_max;
    param.planner_w_max                   = config.planner_w_max;

    if( frame2frame = config.sim_pause ){
        if( frame2frame_switch_old != config.sim_step ){ frame2frame_callback = true; }
    }
    frame2frame_switch_old = config.sim_step;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
