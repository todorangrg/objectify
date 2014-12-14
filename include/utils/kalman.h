#ifndef KALMAN_H
#define KALMAN_H
#include "opencv/cv.h"
#include "utils/base_classes.h"

class NeighData;

class ObjMat{
public:
    int i_min;
    cv::Mat S_O;
    cv::Mat P_OO;
};

class KalmanSLDM{
public:

    bool pos_init;

    SegmentDataPtrVectorPtr    seg_init;
    SegmentDataExtPtrVectorPtr seg_ext;
    ros::Time                  time_stamp;
    std::map<ObjectDataPtr, ObjMat> Oi;
    cv::Mat S;
    cv::Mat S_bar;
    cv::Mat P;

    void init      (RState rob_x);
    void prediction(SegmentDataPtrVectorPtr &input, KInp &u);
    void advance   (InputData & input, bool advance);
    void run       (InputData & input,
                    std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >  & neigh_data_oe,
                    std::map <SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_ne,
                    std::map <SegmentDataPtr   , std::vector<NeighDataInit> > & neigh_data_oi,
                    std::map <SegmentDataPtr   , std::vector<NeighDataInit> > & neigh_data_ni);

    RState rob_x_now(){return RState(S);}
    RState rob_x_bar(){return RState(S_bar);}
    RState rob_x_old(){return RState(S_old);}

    //Constructors & Destructors
    KalmanSLDM(RecfgParam& _param, SensorTf& _tf_sns);
    ~KalmanSLDM(){}
private:
    static const int rob_param   = 3;
    static const int obj_param   = 9;
    static const int input_param = 2;
    static const int z_param     = 3;

    double  v_static;
    double  w_static;

    cv::Mat P_RO;
    cv::Mat P_OR;
    cv::Mat P_OO;

    SegmentDataPtrVectorPtr    seg_init_old;
    std::map<ObjectDataPtr, ObjMat> Oi_old;
    cv::Mat S_old;
    cv::Mat S_bar_old;
    cv::Mat P_old;

    //kalman_base ---
    int  assign_unique_obj_id();
    void update_sub_mat();
    bool add_obj(ObjectDataPtr seg, KObjZ kObjZ);
    bool rmv_obj(ObjectDataPtr seg);
    cv::Mat  Fxi(ObjectDataPtr seg);
    // ---kalman_base

    //kalman_prediction ---
    void predict_rob    (RState  rob_f0, KInp u, cv::Mat& Gt_R, cv::Mat& Q);
    void predict_obj    (KInp u, cv::Mat &Gt, cv::Mat& Q);
    void predict_p_cloud(SegmentDataPtrVectorPtr &input, RState  rob_f0, KInp u);
    cv::Mat Gt_Oi(double dt);
    cv::Mat Q_Oi (double _obj_alfa_xy, double _obj_alpha_phi, double dt);
    // ---kalman_prediction

    //kalman_update ---
    void init_Oi  (ObjectDataPtr obj, xy obj_com_bar_f1, double dt);
    void update_Oi(ObjectDataPtr seg, KObjZ kObjZ);
    // ---kalman_update

    //kalman_update ---
    void extract_common_pairs   (std::vector<ObjectDataPtr>                               &    o_comm,
                                 std::vector<CorrInput>                                   & list_comm,
                                 std::map<SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_oe,
                                 std::map<SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_ne);
    void propagate_no_update_obj(std::map <SegmentDataPtr  , std::vector<NeighDataInit> > & neigh_data_oi,
                                 std::map <SegmentDataPtr  , std::vector<NeighDataInit> > & neigh_data_ni, double _dt);
    bool compute_avg_miu_sigma(std::vector<CorrInput> & list_comm, KObjZ & avg);
    void propag_extr_p_clouds (std::vector<CorrInput> & list_comm, std::map<ObjectDataPtr  , ObjMat>::iterator                        oi);
    void add_new_obj          (SegmentDataPtrVectorPtr & input   , std::map <SegmentDataPtr, std::vector<NeighDataInit> >& neigh_data_ni);
    void remove_lost_obj();
    // ---kalman_update

    double&   rob_alfa_1;
    double&   rob_alfa_2;
    double&   rob_alfa_3;
    double&   rob_alfa_4;
    double&   obj_alfa_xy_min;
    double&   obj_alfa_xy_max;
    double&   obj_alfa_max_vel;
    double&   obj_alfa_phi;
    double&   obj_init_pow_dt;
    double&   obj_timeout;
    double&   discard_old_seg_perc;
    double&   no_upd_vel_hard0;

    SensorTf& tf_sns;
};

#endif // KALMAN_H
