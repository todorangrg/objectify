
#include "data_processing/dp_node.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>


#include "data_processing/dp.h"
#include "utils/math.h"
#include "visual/plot_data.h"


int main( int argc , char **argv ) {

    ros::init( argc , argv , "obj" );
    ros::NodeHandle n;
    RecfgParam dyn_param;
    SensorTf sensor_tf;
    PlotData  plot_data("Object_tracking_visualization",dyn_param, sensor_tf);
    PlotConv  plot_conv("Convolution_visualization",dyn_param);
    DataProcessingNode dp(n, dyn_param, sensor_tf, plot_data,plot_conv);
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

        std::stringstream info; info.str(""); info<<"[ms]Full cycle time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot_data.putInfoText(info,0,plot_data.black);//does not freeze value in step sim mode

        plot_data.update();
        plot_conv.update();

        bool update = ( dp.frame2frame && dp.frame2frame_callback ) || ( !dp.frame2frame );
        if(update){
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

DataProcessingNode::DataProcessingNode( ros::NodeHandle & n, RecfgParam& param, SensorTf& sns_tf, PlotData& plot, PlotConv& plot_conv ):
        DataProcessing( param, sns_tf, plot,plot_conv ), n_( n ),
        odom_sub ( n , "/r1/odom"            , 10 ),/*base_pose_ground_truth*/
        laser_sub( n , "/r1/front_laser/scan", 10 ),/*base_scan*/
        sleep_freq(10),
        sync( MySyncPolicy( 10 ) , odom_sub , laser_sub ),
        new_data(false){

    sync.registerCallback( boost::bind( &DataProcessingNode::callback_odom_laser , this , _1 , _2 ) );
    reconfigureFnc_ =      boost::bind( &DataProcessingNode::callbackParameters  , this , _1 , _2 );
    reconfigureServer_.setCallback( reconfigureFnc_ );

    pause_gazebo  =  n.serviceClient<std_srvs::Empty>("gazebo/pause_physics");
    unpause_gazebo = n.serviceClient<std_srvs::Empty>("gazebo/unpause_physics");
    //unpause_gazebo.call(empty_srv);
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

    param.kalman_rob_alfa_1               = config.kalman_rob_alfa_1;
    param.kalman_rob_alfa_2               = config.kalman_rob_alfa_2;
    param.kalman_rob_alfa_3               = config.kalman_rob_alfa_3;
    param.kalman_rob_alfa_4               = config.kalman_rob_alfa_4;
    param.kalman_obj_alfa_xy_min          = config.kalman_obj_alfa_xy_min;
    param.kalman_obj_alfa_xy_max          = config.kalman_obj_alfa_xy_max;
    param.kalman_obj_alfa_max_vel         = config.kalman_obj_alfa_max_vel;
    param.kalman_obj_alfa_phi             = config.kalman_obj_alfa_phi;
    param.kalman_obj_init_pow_dt          = config.kalman_obj_init_pow_dt;
    param.kalman_obj_timeout              = config.kalman_obj_timeout;
    param.kalman_discard_old_seg_perc     = config.kalman_discard_old_seg_perc;

    if( frame2frame = config.sim_pause ){
        if( frame2frame_switch_old != config.sim_step ){ frame2frame_callback = true; }
    }
    frame2frame_switch_old = config.sim_step;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void DataProcessingNode::callback_odom_laser(const nav_msgs::OdometryConstPtr &_odom , const sensor_msgs::LaserScanConstPtr &_laser ){
    //new_data= true;
    RState stateOdom(_odom->pose.pose.position.x, _odom->pose.pose.position.y, quaternion2Angle2D(_odom->pose.pose.orientation));

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
    vel.v += noise.normalDist(0, sim_rob_alfa_1 * sqr(vel.v) + sim_rob_alfa_2 * sqr(vel.w));
    vel.w += noise.normalDist(0, sim_rob_alfa_3 * sqr(vel.v) + sim_rob_alfa_4 * sqr(vel.w));//noising the input

    callback_odom_laser_data = InputData(laser_raw_now, stateOdom, vel, _odom->header.stamp.now());
    param.cb_sensor_point_angl_inc=_laser->angle_increment;
    param.cb_sensor_point_angl_max=_laser->angle_max;
    param.cb_sensor_point_angl_min=_laser->angle_min;
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

