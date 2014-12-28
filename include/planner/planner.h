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


    template <class SegData>
    void potential_weight(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_r2n, int frame, xy & full_pot);
    template <class SegData>
    void tangent_bug(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_r2n, int frame, RState rob_pos,
                     double d_followed_old, ObjectDataPtr & o_followed_old, double & d_followed_new, ObjectDataPtr & o_followed_new);
    void run(double _pred_time);

    //Constructors & Destructors
    TangentBug(RecfgParam& param, SensorTf& _tf_sns, KalmanSLDM & _k, Segmentation & _segmentation);
    ~TangentBug(){}
private:

    void vel_controller(xy & full_pot, polar target);
    void find_dir_followed(int frame, int   dir_followed_old, ObjectDataPtr o_followed_old,
                                      int & dir_followed    , ObjectDataPtr o_followed    , double d_followed, polar & target);

    SensorTf   & tf_sns;
    KalmanSLDM & k;
    double     & sensor_range_max;
    double     & angle_max;
    double     & angle_min;
    Segmentation & segmentation;

    double     & pot_scale;
    double     & w_kp_goal;
    double     & v_kp_w;
    double     & v_kp_goal;
    double     & v_max;
    double     & w_max;
};

#endif // PLANNER_H
