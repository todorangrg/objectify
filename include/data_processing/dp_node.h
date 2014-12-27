
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

    Distributions nd;

    double sim_rob_alfa_1;
    double sim_rob_alfa_2;
    double sim_rob_alfa_3;
    double sim_rob_alfa_4;

    double quaternion2Angle2D ( const geometry_msgs::Quaternion &quat );


};

#endif // DP_NODE_H
