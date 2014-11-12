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



///TODO DISCARD AND USE TF LIBRARY
void SensorTf::init() {
    cv::Mat_<double> R = (cv::Mat_<double>(3,3) <<  1, 0, 0,
                                                    0, 1, 0,
                                                    0, 0, 1);    // Rotate
    cv::Mat_<double> T = (cv::Mat_<double>(3,3) <<  1, 0, 0.225,
                                                    0, 1, 0,
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
void FrameTf::init(SensorFrame &laser_new, SensorFrame &laser_old){
    double c1 = cos( - laser_new.rob_x.angle), s1 = sin( - laser_new.rob_x.angle);
    double c2 = cos(   laser_old.rob_x.angle), s2 = sin(   laser_old.rob_x.angle);
    double xx = laser_old.rob_x.x - laser_new.rob_x.x;
    double yy = laser_old.rob_x.y - laser_new.rob_x.y;
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

SegmentDataExt& SegmentDataExt::operator= (const SegmentDataExt& seg){
  return *this;
}

SegmentData::SegmentData(boost::shared_ptr<ObjectData> _parrent, int _id) :
    parrent(_parrent),
    id(_id){}

SegmentDataExt::SegmentDataExt(const boost::shared_ptr<SegmentData> _parrent) :
    parrent(_parrent),
    id(_parrent->id){}

SegmentDataExt::SegmentDataExt(const SegmentDataExt& _seg) :
    id( _seg.id),
    parrent(_seg.parrent),
    len(_seg.len),
    corr_flag(CORR_NEWOBJ){}

SegmentDataExt::SegmentDataExt(const SegmentDataExtPtrVectorIter& _seg) :
    id((*_seg)->id),
    parrent((*_seg)->parrent),
    corr_flag(CORR_NEWOBJ),
    len((*_seg)->len){}

SegmentDataExt::SegmentDataExt(const SegmentDataExtPtrVectorIter& _seg, int _id) :
    id( _id),
    parrent((*_seg)->parrent),
    corr_flag(CORR_NEWOBJ),
    len((*_seg)->len){}


CorrInput::CorrInput(SegmentDataExtPtr _frame_old, SegmentDataExtPtr _frame_new, double _stitch_perc, bool _reverse) :
    frame_old(_frame_old),
    frame_new(_frame_new),
    stitch_perc(_stitch_perc),
    reverse(_reverse){}

SensorFrame::SensorFrame(State _rob_x,Command _rob_vel, ros::Time _time_stamp,PointDataVectorPtr &input) :
    rob_x(_rob_x),
    rob_vel(_rob_vel),
    time_stamp(_time_stamp),
    sensor_raw(input),
    seg_init(new SegmentDataPtrVector   ),
    seg_ext (new SegmentDataExtPtrVector){}

SensorData::SensorData(RecfgParam& _param) :
    sensor_data_catched_cycles(_param.sensor_data_catched_cycles),
    true_rob_pos(_param.true_rob_pos),
    status(NO_FRAME){
    objects = boost::shared_ptr<std::vector<boost::shared_ptr<ObjectData> > >(new std::vector<boost::shared_ptr<ObjectData> >);
}

void  SensorData::push_back_frame( SensorFrame _new, KalmanSLDM& k ){
    if( _new.sensor_raw == NULL ){
        return;
    }
    status = NEW_FRAME;
    sensor_history.push_back(_new);
    frame_new = --sensor_history.end();

    while( sensor_history.size() > sensor_data_catched_cycles ){
        frame_old++;
        sensor_history.pop_front();
    }
    sensor_history.front().past_time = sensor_history.back().time_stamp-sensor_history.front().time_stamp;
    if ( sensor_history.size() == 2 ){
        frame_old= sensor_history.begin();


        for(int i=0;i < frame_old->seg_init->size(); i++){//oldening the new frame
            frame_old->seg_init->at(i)->parrent->fr_stat = FRAME_OLD;
        }
        if(k.pos_init){
            sensor_history.front().rob_x.x = k.S.at<double>(0,0);
            sensor_history.front().rob_x.y = k.S.at<double>(1,0);
            sensor_history.front().rob_x.angle = k.S.at<double>(2,0);
        }

        double alpha_motion_[6] = {0.5};
        double a1 = 0.1;//alpha_motion_[0];
        double a2 = 0.05;//alpha_motion_[1];
        double a3 = 0.05;//alpha_motion_[2];
        double a4 = 0.1;//alpha_motion_[3];
        double a5 = 0.1;//alpha_motion_[4];
        double a6 = 0.1;//alpha_motion_[5];
        double a_max = 0.5;


        Distributions noise;
        double dt = sensor_history.front().past_time.toSec();
        sensor_history.back().rob_vel.v += noise.normalDist(0, a1 * sqr(sensor_history.back().rob_vel.v) + a2 * sqr(sensor_history.back().rob_vel.w));
        sensor_history.back().rob_vel.w += noise.normalDist(0, a3 * sqr(sensor_history.back().rob_vel.v) + a4 * sqr(sensor_history.back().rob_vel.w));//noising the input

//        if((sensor_history.back().rob_vel.v > sensor_history.front().rob_vel.v + a_max * dt)||
//           (sensor_history.back().rob_vel.v < sensor_history.front().rob_vel.v - a_max * dt)){
//            sensor_history.back().rob_vel.v = sensor_history.front().rob_vel.v - sgn(sensor_history.front().rob_vel.v - sensor_history.back().rob_vel.v) * a_max * dt;
//        }
//        if((sensor_history.back().rob_vel.w > sensor_history.front().rob_vel.w + a_max /( M_PI) * dt)||
//           (sensor_history.back().rob_vel.w < sensor_history.front().rob_vel.w - a_max /( M_PI) * dt)){
//            sensor_history.back().rob_vel.w = sensor_history.front().rob_vel.w - sgn(sensor_history.front().rob_vel.w - sensor_history.back().rob_vel.w) * a_max /( M_PI) * dt;
//        }

        State s_new, s_old = sensor_history.front().rob_x;
        Command cmd_new; cmd_new.set( (sensor_history.back().rob_vel.v /*+ sensor_history.front().rob_vel.v*/) /*/ 2.0*/,
                                      (sensor_history.back().rob_vel.w /*+ sensor_history.front().rob_vel.w*/) /*/ 2.0*/);
        double vv=cmd_new.v;
        double ww=cmd_new.w;
        double ss=noise.normalDist(0, a5 *cmd_new.v*cmd_new.v + a6 *cmd_new.w*cmd_new.w);
        if(fabs(ww) > 0.001){
            double v_w=vv / ww;
            s_new.x = s_old.x - v_w * sin(s_old.angle) + v_w * sin(s_old.angle + ww * dt);
            s_new.y = s_old.y + v_w * cos(s_old.angle) - v_w * cos(s_old.angle + ww * dt);
        }
        else{
            s_new.x = s_old.x + vv * dt * cos(s_old.angle);
            s_new.y = s_old.y + vv * dt * sin(s_old.angle);

        }
        s_new.angle = s_old.angle + ww * dt + ss * dt;
        if(!true_rob_pos){
            sensor_history.back().rob_x = s_new;
        }


        status = OLD_FRAME;
    }
    if( sensor_history.size() >= 2 ){
        tf_frm.init(*frame_new,*frame_old);
    }
}

void SensorData::push_virtual_time(double virtual_time){
    if(sensor_history.size() == sensor_data_catched_cycles){
        frame_old->past_time =ros::Duration( (sensor_data_catched_cycles-1) * virtual_time );
    }
}

