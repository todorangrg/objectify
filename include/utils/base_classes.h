#ifndef BASE_CLASSES_H
#define BASE_CLASSES_H

#include <ros/ros.h>
#include <vector>
#include <list>
#include "boost/shared_ptr.hpp"
#include "boost/variant.hpp"
#include "opencv/cv.h"
#include "utils/math.h"

class PointData;
class PointDataSample;
class SegmentData;
class SegmentDataExt;
class SegmentDataNeigh;
template<class SegData>
class IteratorIndexSet;
class SensorFrame;
class ObjectData;
class KalmanSLDM;
class TFdata;

typedef std::vector<PointData>                     PointDataVector;
typedef boost::shared_ptr<PointDataVector>         PointDataVectorPtr;
typedef PointDataVector::iterator                  PointDataVectorIter;

typedef std::vector<PointDataSample>               PointDataSampleVector;
typedef boost::shared_ptr<PointDataSampleVector>   PointDataSampleVectorPtr;


typedef boost::shared_ptr<SegmentData>             SegmentDataPtr;
typedef std::vector<SegmentDataPtr>                SegmentDataPtrVector;
typedef boost::shared_ptr<SegmentDataPtrVector>    SegmentDataPtrVectorPtr;
typedef SegmentDataPtrVector::iterator             SegmentDataPtrVectorIter;
//


typedef std::vector<SegmentDataExt>                SegmentDataExtVector;
typedef boost::shared_ptr<SegmentDataExtVector>    SegmentDataExtVectorPtr;
typedef SegmentDataExtVector::iterator             SegmentDataExtVectorIter;

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
    CORR_121 = 0,
    CORR_12MANY = 1
};

enum ConvolStatus{
    CONV_REF,
    CONV_SPL
};

enum FrameStatus{
    FRAME_NEW,
    FRAME_OLD
};

enum CorrPairFlag{
    CORR_NO_CORR_NEW,
    CORR_NO_CORR_OLD,
    CORR_SINGLE2,
    CORR_SINGLE1MULTI1,
    CORR_MULTI2
};

enum CorrList{
    INIT,
    EXT
};



class RecfgParam{
public:

    bool viz_data;
    bool viz_data_grid;
    bool viz_data_raw;
    bool viz_data_preproc;
    bool viz_data_oult_preproc;
    bool viz_data_segm_init;
    bool viz_data_segm_ext;

    double sensor_r_max;
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
    double convol_min_len_perc;
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

    double kalman_rob_alfa_1;
    double kalman_rob_alfa_2;
    double kalman_rob_alfa_3;
    double kalman_rob_alfa_4;
    double kalman_obj_alfa_xy;
    double kalman_obj_alfa_phi;
    double kalman_obj_timeout;

    std::list<double> smooth_mask;
    void init_normal_smooth_mask();

    double cb_sensor_point_angl_inc;//USED FOR ANGULAR RESOLUTION
    double cb_sensor_point_angl_max;
    double cb_sensor_point_angl_min;

    //Constructors & Destructors
    RecfgParam(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class RState{
public:

    double xx; double xy; double xphi;
    RState(cv::Mat _S);
    RState(double _x, double _y, double _ang) : xx(_x),xy(_y),xphi(_ang){}
    RState(){}
};

class KInp {
public:

    double v;  /// vorwaerts geschwindigkeit m/sec
    double w;  /// winkel geschwindigkeit in rad/sec
    double dt;
    KInp(double _v, double _w, double _dt) : v(_v),w(_w),dt(_dt){}
    KInp(double _v, double _w) : v(_v),w(_w){}
    KInp(){}
};

class OiState{
public:

    double xx; double xy; double xphi;
    double vx; double vy; double vphi;
    double ax; double ay; double aphi;
    OiState(cv::Mat _S_O);
    OiState(){}
};

class KObjZ{
public:

    xy     pos;
    double phi;
    cv::Matx33d Q;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class FrameTf{///TODO DISCARD AND USE TF LIBRARY
public:

    void init(RState rob_x, RState rob_bar_x);
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
    xy     getXY() {return    xy(Ms2r.at<double>(0,2), Ms2r.at<double>(1,2));}
    double getPhi(){return atan2(Ms2r.at<double>(1,0), Ms2r.at<double>(0,0));}
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

class PointData{
public:

    double                          r;
    double                          angle;
    boost::weak_ptr<SegmentDataExt> child;
    boost::weak_ptr<SegmentData>    neigh;

    //Constructors & Destructors
    PointData()       : r(0.0), angle(0.0)    {}
    PointData(polar p): r(p.r), angle(p.angle){}
    //~PointData(){}
};

class PointDataCpy{
public:

    double         r;
    double         angle;
    PointData      p_parrent;
    SegmentDataPtr s_parrent;

    //Constructors & Destructors
    PointDataCpy(PointData p,SegmentDataPtr _s_parrent): r(p.r), angle(p.angle), p_parrent(p), s_parrent(_s_parrent){}
    ~PointDataCpy(){}
};

class PointDataSample : public PointData{
public:

    int*   no;
    double normal_ang;
    double fade_out;

    //Constructors & Destructors
    PointDataSample(PointData p)                                           : PointData(p), no(NULL), normal_ang(0.0)       , fade_out(0.0)     {}
    PointDataSample(PointData p,int* no)                                   : PointData(p), no(no)  , normal_ang(0.0)       , fade_out(0.0)     {}
    PointDataSample(PointData p,int* no,double normal_ang)                 : PointData(p), no(no)  , normal_ang(normal_ang), fade_out(0.0)     {}
    PointDataSample(PointData p,int* no,double normal_ang, double fade_out): PointData(p), no(no)  , normal_ang(normal_ang), fade_out(fade_out){}
    ~PointDataSample(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class TfVar{
public:

    xy             com;     // icp com of parrent (xy), angle = 0
    xy             com_tf;
    cv::Matx33d    T;       // tf parrent -> neighbour
    cv::Matx33d    Q;  // covariance of tf-ed parrent icp com
    double         len;

    //Constructors & Destructors
    TfVar(xy _com, xy _com_tf, cv::Matx33d _T, cv::Matx33d _Q, double _len):com(_com), com_tf(_com_tf), T(_T), Q(_Q), len(_len){}
    TfVar(){}
    ~TfVar(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class TFdata{
public:

    SegmentDataExtPtr seg;      // segment of neighbour
    ConvolStatus     conv_stat; // link status
    double           score;     // link score
    TfVar            tf;        // direct tf  (from this to neighbour)
    TfVar            tf_inv;    // inverse tf (from neighbour to this)

    //Constructors & Destructors
    TFdata(SegmentDataExtPtr _seg, ConvolStatus   _conv_stat, double _score):seg(_seg),conv_stat(_conv_stat),score(_score){}
    TFdata(SegmentDataExtPtr _seg, ConvolStatus   _conv_stat, double _score, TfVar _tf, TfVar _tf_inv):seg(_seg),conv_stat(_conv_stat),score(_score),tf(_tf),tf_inv(_tf_inv){}
    TFdata(){}
    ~TFdata(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class ConvData{
public:

    boost::shared_ptr<std::vector<PointDataSample> > p_cd; // const dist sampled points
    xy                                               com;  // resulting com of all segment
    boost::shared_ptr<std::vector<TFdata> >          tf;   // tf data towards neighbours
    SegmentDataExtPtr                                seg;  // segment adress

    ConvData(SegmentDataExtPtr _seg){
        seg  = _seg;
        p_cd = boost::shared_ptr<std::vector<PointDataSample> >(new std::vector<PointDataSample>);
        tf   = boost::shared_ptr<std::vector<TFdata> >         (new std::vector<TFdata>);
    }

    //Constructors & Destructors
    ~ConvData(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class ObjectData{
public:

    int  id;
    bool solved;

    //Constructors & Destructors
    ObjectData(int _id): id(_id), solved(true){}
    ~ObjectData(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SegmentDataBase{
public:

    int             id;//only for debug
    PointDataVector p;

    CorrFlag       corr_flag;
    bool           solved;
    bool           is_old;

    ObjectDataPtr getObj()                  {return obj;}
    void          setObj(ObjectDataPtr _obj){obj = _obj;}
    xy            getCom()                  {return com;}
    void          setCom(xy _com)           {com = _com;}
    double        getLen()                  {return len;}
    void          setLen(double _len)       {len = _len;}

    //Constructors & Destructors
    SegmentDataBase(                    int _id)                                    :            id(_id), com( 0 ) , len(0)   , corr_flag(CORR_121), solved(false), is_old(false){}
    SegmentDataBase(ObjectDataPtr _obj, int _id)                                    : obj(_obj), id(_id), com( 0 ) , len(0)   , corr_flag(CORR_121), solved(false), is_old(false){}
    SegmentDataBase(ObjectDataPtr _obj, int _id, xy _com, double _len)              : obj(_obj), id(_id), com(_com), len(_len), corr_flag(CORR_121), solved(false), is_old(false){}
    SegmentDataBase(ObjectDataPtr _obj, int _id, xy _com, double _len, CorrFlag _cf): obj(_obj), id(_id), com(_com), len(_len), corr_flag(_cf)     , solved(false), is_old(false){}
    ~SegmentDataBase(){}
protected:

    ObjectDataPtr obj;
    xy            com;
    double        len;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SegmentData : public SegmentDataBase{
public:

    ObjectDataPtr getParrent()                  {return obj;}
    void          setParrent(ObjectDataPtr _obj){obj = _obj;}

    //Constructors & Destructors
    SegmentData(int _id)                                         : SegmentDataBase(                           _id){}
    SegmentData(ObjectDataPtr _obj, int _id)                     : SegmentDataBase(_obj             ,         _id){}
    SegmentData(ObjectDataPtr _obj, int _id, SegmentDataPtr _seg): SegmentDataBase(_obj             ,         _id,    _seg->com,    _seg->len){}
    SegmentData(SegmentDataPtrVectorIter _seg)                   : SegmentDataBase((*_seg)->getObj(), (*_seg)->id, (*_seg)->com, (*_seg)->len){}
    SegmentData(SegmentDataPtrVectorIter _seg, int _id)          : SegmentDataBase((*_seg)->getObj(),         _id, (*_seg)->com, (*_seg)->len){}
    SegmentData(SegmentDataPtr           _seg         )          : SegmentDataBase(   _seg->getObj(),    _seg->id,    _seg->com,    _seg->len){}
    ~SegmentData(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class SegmentDataExt : public SegmentDataBase{
public:

    boost::shared_ptr<ConvData> conv;
    double len_init_neg;
    double len_init_pos;

    SegmentDataPtr getParrent()                       {return parrent    ;}
    void           setParrent(SegmentDataPtr _parrent){parrent = _parrent;}

    //Constructors & Destructors
    SegmentDataExt(SegmentDataExt& _segext)                     : SegmentDataBase(    _segext.getObj(),     _segext.id,     _segext.com, _segext.len    , _segext.corr_flag    ), parrent(_segext.getParrent())    , len_init_neg(_segext.len_init_neg), len_init_pos(_segext.len_init_pos){}
    SegmentDataExt(SegmentDataPtr _seg, int _id)                : SegmentDataBase(      _seg->getObj(),            _id,         xy(0,0),               0, CORR_121             ), parrent(_seg)                    , len_init_neg(0), len_init_pos(0){}
    SegmentDataExt(SegmentDataExtPtrVectorIter _segext)         : SegmentDataBase((*_segext)->getObj(), (*_segext)->id, (*_segext)->com, (*_segext)->len, (*_segext)->corr_flag), parrent((*_segext)->getParrent()), len_init_neg((*_segext)->len_init_neg), len_init_pos((*_segext)->len_init_pos){}
    SegmentDataExt(SegmentDataExtPtrVectorIter _segext, int _id): SegmentDataBase((*_segext)->getObj(),            _id, (*_segext)->com, (*_segext)->len, (*_segext)->corr_flag), parrent((*_segext)->getParrent()), len_init_neg((*_segext)->len_init_neg), len_init_pos((*_segext)->len_init_pos){}
    SegmentDataExt& operator= (const SegmentDataExt & seg);
    ~SegmentDataExt(){}
private:

    SegmentDataPtr parrent;
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

template<class SegData>
void SegCopy(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > & from, boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > & to){
    to = boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > ( new std::vector<boost::shared_ptr<SegData> >);
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = from->begin(); ss != from->end(); ss++){
        to->push_back(boost::shared_ptr<SegData>(new  SegData(ss)));
        for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
            to->back()->p.push_back(*pp);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
//TODO template it
class NeighDataInit{
public:

    SegmentDataPtr neigh;
    double         prob_fwd;
    double         prob_rev;
    boost::shared_ptr<std::vector<TfVar> > tf;
    bool           has_tf;

    //Constructors & Destructors
    NeighDataInit(SegmentDataPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev), has_tf(false), tf(new std::vector<TfVar>){}
    ~NeighDataInit(){}
};

class NeighDataExt{
public:

    SegmentDataExtPtr neigh;
    double            prob_fwd;
    double            prob_rev;
    TfVar             tf;
    bool              has_tf;

    //Constructors & Destructors
    NeighDataExt(SegmentDataExtPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev), has_tf(false){}
    ~NeighDataExt(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class CorrInput{
public:

    SegmentDataExtPtr frame_old;
    SegmentDataExtPtr frame_new;
    double            stitch_perc;
    bool              reverse;

    //Constructors & Destructors
    CorrInput(SegmentDataExtPtr _frame_old,SegmentDataExtPtr _frame_new,double _stitch_perc,bool _reverse);
    ~CorrInput(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class InputData{
public:

    bool                       is_valid;
    ros::Time                  time_stamp;
    RState                     rob_x;
    KInp                       u;

    PointDataVectorPtr         sensor_raw;
    PointDataVectorPtr         sensor_filtered;
    SegmentDataPtrVectorPtr    seg_init;
    SegmentDataExtPtrVectorPtr seg_ext;

    //Constructors & Destructors
    InputData(PointDataVectorPtr &input, RState _rob_x, KInp _u, ros::Time _time_stamp);
    InputData():is_valid(false){}
    ~InputData(){}
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

#endif // BASE_CLASSES_H
