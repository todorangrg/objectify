#ifndef BASE_CLASSES_H
#define BASE_CLASSES_H

#include <ros/ros.h>
#include <vector>
#include <list>
#include "boost/shared_ptr.hpp"
#include "opencv/cv.h"
#include "utils/math.h"

class PointData;
class PointDataSample;
class SegmentData;
class SegmentDataExt;
class SegmentDataNeigh;
class IteratorIndexSet;
class SensorFrame;
class ObjectData;
class KalmanSLDM;

typedef std::vector<PointData>                   PointDataVector;
typedef boost::shared_ptr<PointDataVector>       PointDataVectorPtr;
typedef PointDataVector::iterator                PointDataVectorIter;

typedef std::vector<PointDataSample>             PointDataSampleVector;
typedef boost::shared_ptr<PointDataSampleVector> PointDataSampleVectorPtr;


typedef boost::shared_ptr<SegmentData>          SegmentDataPtr;
typedef std::vector<SegmentDataPtr>             SegmentDataPtrVector;
typedef boost::shared_ptr<SegmentDataPtrVector> SegmentDataPtrVectorPtr;
typedef SegmentDataPtrVector::iterator          SegmentDataPtrVectorIter;
//


typedef std::vector<SegmentDataExt>             SegmentDataExtVector;
typedef boost::shared_ptr<SegmentDataExtVector> SegmentDataExtVectorPtr;
typedef SegmentDataExtVector::iterator          SegmentDataExtVectorIter;

//
typedef boost::shared_ptr<SegmentDataExt>          SegmentDataExtPtr;
typedef std::vector<SegmentDataExtPtr>             SegmentDataExtPtrVector;
typedef boost::shared_ptr<SegmentDataExtPtrVector> SegmentDataExtPtrVectorPtr;
typedef SegmentDataExtPtrVector::iterator          SegmentDataExtPtrVectorIter;




typedef std::vector<SegmentDataNeigh>                 SegmentDataNeighVector;
typedef std::vector<SegmentDataNeigh>::iterator       SegmentDataNeighVectorIter;

typedef boost::shared_ptr<ObjectData>                 ObjectDataPtr;

///------------------------------------------------------------------------------------------------------------------------------------------------///

enum CorrFlag{
    CORR_121,
    CORR_12MANY,
    CORR_MANY21,
    CORR_NOINNV,
    CORR_NEWOBJ
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

enum SensorDataStatus{
    NO_FRAME,
    NEW_FRAME,
    OLD_FRAME
};

enum FrameStatus{
    FRAME_NEW,
    FRAME_OLD
};

enum ConvolStatus{
    CONV_REF,
    CONV_SPL
};



///------------------------------------------------------------------------------------------------------------------------------------------------///

class RecfgParam{
public:

    bool viz_data;
    bool viz_data_grid;
    int  viz_data_res;
    bool viz_data_raw;
    bool viz_data_preproc;
    bool viz_data_oult_preproc;
    bool viz_data_segm;


    int sensor_data_catched_cycles;
    bool true_rob_pos;
    double sensor_range_max;
    double sensor_noise_sigma;

    bool   preproc;
    bool   preproc_filter;
    double preproc_filter_circle_rad;      //SHOULD BE ANGULAR RESOLUTION DEPENDANT
    bool   preproc_filter_circle_rad_scale;//TRUE AS DEFAULT WOULD BE COOL, BUT TEST RUN SPEED FOR DIFFRENT RESOLUTIONS
    double preproc_filter_sigma;

    bool   segm;
    double segm_outl_circle_rad;//SHOULD BE ANGULAR RESOLUTION DEPENDANT?
    double segm_outl_sigma;
    double segm_outl_prob_thres;//NO IDEEA IF NEEDED
    double segm_discont_dist;//GOOD

    bool   corr;
    bool   viz_data_corr_links;
    double corr_queue_d_thres;   //SHOULD THIS BE IN PERCENT?
    double corr_neigh_circle_rad;  //COOL TO USE THIS BEFORE SEGMENTING D(T) AND THEN SEGMENTING WITH SNAPPED PREDICTED INVERSE SPEEDS


    bool viz_convol;
    bool viz_data_tf;
    bool convol_full_search;
    bool viz_convol_all;
    bool viz_convol_normals;
    bool viz_convol_tf;
    bool viz_convol_tf_ref2spl;
    int  viz_correl_queue_no;
    int  viz_convol_step_no;

    bool convol_SVD;

    double convol_sample_dist;//SHOULD BE VARIABLE WITH OBJECT INFORMATION
    double convol_marg_extr_excl;
    int    convol_normals_smooth_mask_size;
    double convol_normals_smooth_mask_dist;
    double convol_key_d_angle;


    double convol_com_dr_thres;
    double convol_ang_mean_thres;
    double convol_ang_var_thres;
    double convol_sqr_err_thres;
    double convol_p_no_perc_thres;
    double convol_score_thres;



    std::list<double> smooth_mask;
    void init_normal_smooth_mask();

    double cb_sensor_point_angl_inc;//USED FOR ANGULAR RESOLUTION
    double cb_sensor_point_angl_max;
    double cb_sensor_point_angl_min;

    //Constructors
    RecfgParam(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class FrameTf{///TODO DISCARD AND USE TF LIBRARY
public:
    void init(SensorFrame &laser_new, SensorFrame &laser_old);
    xy rn2ro(double _x, double _y);
    xy ro2rn(double _x, double _y);
    xy rn2ro(const xy &_p) {
        return rn2ro(_p.x, _p.y);
    }
    xy ro2rn(const xy &_p) {
        return ro2rn(_p.x, _p.y);
    }
private:
    cv::Mat_<double> Mrn2ro;
    cv::Mat_<double> Mro2rn;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SensorTf{///TODO DISCARD AND USE TF LIBRARY
public:
    void init();
    xy s2r(double _x, double _y);
    xy r2s(double _x, double _y);
    xy s2r(const xy &_p) {
        return s2r(_p.x, _p.y);
    }
    xy r2s(const xy &_p) {
        return r2s(_p.x, _p.y);
    }
    SensorTf(){init();}
private:
    cv::Mat_<double> Mr2s;
    cv::Mat_<double> Ms2r;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

struct State {
    double x;  /// x position
    double y;  /// y position
    double angle;  /// angle
    void set(double _x, double _y, double _angle) { x = _x, y = _y,  angle = _angle; }
    xy position() { return xy(x,y); }
};

struct Command {
    double v;  /// vorwaerts geschwindigkeit m/sec
    double w;  /// winkel geschwindigkeit in rad/sec
    void set(double _v, double _w) {
        v = _v, w = _w;
    };
};


class KControl{
public:
    double v;
    double w;
    double dt;
    KControl(Command _u, double _dt):v(_u.v),w(_u.w),dt(_dt){}
    KControl(double _v, double _w, double _dt):v(_v),w(_w),dt(_dt){}
};

class KObjZ{
public:
    xy     pos;
    double phi;
    cv::Matx33d Q;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class PointData{
public:
    double r;
    double angle;

    //Constructors
    PointData()       : r(0.0), angle(0.0)    {}
    PointData(polar p): r(p.r), angle(p.angle){}
};

class PointDataSample : public PointData{
public:
    int*   no;
    double normal_ang;
    double fade_out;
    PointDataSample(PointData p)                                           : PointData(p), no(NULL), normal_ang(0.0)       , fade_out(0.0)     {}
    PointDataSample(PointData p,int* no)                                   : PointData(p), no(no)  , normal_ang(0.0)       , fade_out(0.0)     {}
    PointDataSample(PointData p,int* no,double normal_ang)                 : PointData(p), no(no)  , normal_ang(normal_ang), fade_out(0.0)     {}
    PointDataSample(PointData p,int* no,double normal_ang, double fade_out): PointData(p), no(no)  , normal_ang(normal_ang), fade_out(fade_out){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///
class TfVar{
public:
    xy             com;     // icp com of parrent (xy), angle = 0
    cv::Matx33d    T;       // tf parrent -> neighbour
    xy             xy_mean; // mean       of tf-ed parrent icp com (xy)
    cv::Matx22d    xy_cov;  // covariance of tf-ed parrent icp com (xy)
    double         ang_mean;// mean       of tf-ed parrent icp com (angle)
    double         ang_cov; // covariance of tf-ed parrent icp com (angle)
    TfVar(xy _com, cv::Matx33d _T, xy _xy_mean, cv::Matx22d _xy_cov, double _ang_mean, double _ang_cov):com(_com),T(_T), xy_mean(_xy_mean), xy_cov(_xy_cov), ang_mean(_ang_mean), ang_cov(_ang_cov)
    {
    }
    TfVar(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///
class TFdata{
public:
    SegmentDataExtPtr seg;    // segment of neighbour
    ConvolStatus   conv_stat; // link status
    double         score;     // link score
    TfVar          tf;        // direct tf  (from this to neighbour)
    TfVar          tf_inv;    // inverse tf (from neighbour to this)
    TFdata(SegmentDataExtPtr _seg, ConvolStatus   _conv_stat, double _score):seg(_seg),conv_stat(_conv_stat),score(_score){}
    TFdata(SegmentDataExtPtr _seg, ConvolStatus   _conv_stat, double _score, TfVar _tf, TfVar _tf_inv):seg(_seg),conv_stat(_conv_stat),score(_score),tf(_tf),tf_inv(_tf_inv){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class ConvData{
public:
    boost::shared_ptr<std::vector<PointDataSample> > p_cd; // const dist sampled points
    xy                                               com;  // resulting com of all segment
    boost::shared_ptr<std::vector<TFdata> >          tf;   // tf data towards neighbours
    SegmentDataExtPtr                                seg;  // segment adress

    ConvData(SegmentDataExtPtr _seg){
        seg = _seg;
        p_cd = boost::shared_ptr<std::vector<PointDataSample> >(new std::vector<PointDataSample>);
        tf   = boost::shared_ptr<std::vector<TFdata> >         (new std::vector<TFdata>);
    }
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class ObjectData{
public:
    FrameStatus fr_stat;
    int         id;
    //Constructors
     ObjectData(FrameStatus _fr_stat, int _id):fr_stat(_fr_stat), id(_id){}
};

class SegmentData{
public:
    int id;
    boost::shared_ptr< ObjectData > parrent;
    PointDataVector p;
    xy com;

    //Constructors
    SegmentData(boost::shared_ptr<ObjectData> _parrent, int _id);
};

class SegmentDataExt{
public:
    int id;
    boost::shared_ptr< SegmentData > parrent;
    PointDataVector p;
    double len;
    boost::shared_ptr<ConvData> conv;

    CorrFlag corr_flag;

    //Constructors
    SegmentDataExt(const boost::shared_ptr<SegmentData> _parrent);
    SegmentDataExt(const SegmentDataExt& _seg);
    SegmentDataExt(const SegmentDataExtPtrVectorIter& _seg);
    SegmentDataExt(const SegmentDataExtPtrVectorIter& _seg, int _id);
    SegmentDataExt& operator= (const SegmentDataExt & seg);
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class CorrInput{
public:
    SegmentDataExtPtr frame_old;
    SegmentDataExtPtr frame_new;
    double stitch_perc;
    bool reverse;

    //Constructors
    CorrInput(SegmentDataExtPtr _frame_old,SegmentDataExtPtr _frame_new,double _stitch_perc,bool _reverse);
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SensorFrame{
public:
    State                rob_x;
    Command              rob_vel;

    ros::Time            time_stamp;
    ros::Duration        past_time;
    PointDataVectorPtr   sensor_raw;
    PointDataVectorPtr   sensor_filtered;

    SegmentDataPtrVectorPtr    seg_init;
    SegmentDataExtPtrVectorPtr seg_ext;

    //Constructors
    SensorFrame(State _rob_x,Command _rob_vel, ros::Time _time_stamp,PointDataVectorPtr &input);
    SensorFrame(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SensorData{
public:
    std::list<SensorFrame>::iterator frame_new;
    std::list<SensorFrame>::iterator frame_old;
    SensorDataStatus status;
    boost::shared_ptr<std::vector<boost::shared_ptr<ObjectData> > > objects;

    void push_back_frame  (SensorFrame _new, KalmanSLDM &k);
    void push_virtual_time(double virtual_time);
    FrameTf& get_tf(){ return tf_frm; }

    //Constructors
    SensorData(RecfgParam& _param);
private:
    int& sensor_data_catched_cycles;
    bool& true_rob_pos;
    std::list<SensorFrame> sensor_history;
    FrameTf tf_frm;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

#endif // BASE_CLASSES_H
