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



#ifndef DP_NODE_H
#define DP_NODE_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>
#include <objectify/objectify_paramConfig.h>

#include "data_processing/dp.h"
#include "utils/math.h"

#include "std_srvs/Empty.h"

class DataProcessingNode : public DataProcessing
{
public:

    ros::ServiceClient pause_gazebo;
    ros::ServiceClient unpause_gazebo;
    std_srvs::Empty    empty_srv;

    ros::Publisher     pub_cmd_;

    bool               new_data;
    double             sleep_freq;
    bool               frame2frame;
    bool               frame2frame_callback;
    double             frame2frame_deltaT;

    InputData          callback_odom_laser_data;

    void callback_odom_laser(const nav_msgs::OdometryConstPtr &_odom, const sensor_msgs::LaserScanConstPtr &_laser);
//     void callback_odom_laser(const geometry_msgs::TwistConstPtr &_odom, const sensor_msgs::LaserScanConstPtr &_laser);
    void callbackParameters ( objectify::objectify_paramConfig &config , uint32_t level );

    //Constructors & Destructors
    DataProcessingNode( ros::NodeHandle & n, RecfgParam& param, SensorTf& sns_tf, PlotData& plot, PlotConv& plot_conv, rosbag::Bag & bag);
    ~DataProcessingNode(){}
private:

    ros::NodeHandle n_;

    dynamic_reconfigure::Server<objectify::objectify_paramConfig>               reconfigureServer_;
    dynamic_reconfigure::Server<objectify::objectify_paramConfig>::CallbackType reconfigureFnc_;

    message_filters::Subscriber  <nav_msgs::Odometry    > odom_sub;
    message_filters::Subscriber  <sensor_msgs::LaserScan> laser_sub;
    typedef message_filters::sync_policies::ApproximateTime< nav_msgs::Odometry, sensor_msgs::LaserScan > MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy>           sync;
    
//     message_filters::Subscriber  <geometry_msgs::Twist    > odom_sub;
//     message_filters::Subscriber  <sensor_msgs::LaserScan> laser_sub;
//     typedef message_filters::sync_policies::ApproximateTime< geometry_msgs::Twist, sensor_msgs::LaserScan > MySyncPolicy;
//     message_filters::Synchronizer<MySyncPolicy>           sync;
    Distributions nd;

    double sim_rob_alfa_1;
    double sim_rob_alfa_2;
    double sim_rob_alfa_3;
    double sim_rob_alfa_4;

    double quaternion2Angle2D ( const geometry_msgs::Quaternion &quat );


};

#endif // DP_NODE_H
