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
    void run(InputData &input, std::map <SegmentDataExtPtr,
             std::vector<NeighDataExt> >& neigh_data_o,
             std::map <SegmentDataExtPtr, std::vector<NeighDataExt > >& neigh_data_n,
             std::map <SegmentDataPtr   , std::vector<NeighDataInit> >& neigh_data_oi);


    void init(RState rob_x);

    void prediction(SegmentDataPtrVectorPtr &input, KInp u);
    void init_Oi(ObjectDataPtr obj, xy obj_com_bar_f1);
    void update_Oi(ObjectDataPtr seg, KObjZ kObjZ);
    void update_Oi_with_Oj(ObjectDataPtr seg, ObjectDataPtr seg_obs);

    SegmentDataPtrVectorPtr    seg_init_old;

    SegmentDataPtrVectorPtr    seg_init_new;
    SegmentDataPtrVectorPtr    seg_init;
    SegmentDataExtPtrVectorPtr seg_ext;
    ros::Time                  time_stamp;
    std::map<ObjectDataPtr, ObjMat> Oi;
    cv::Mat S_R_bar;
    cv::Mat S;
    cv::Mat P;

    RState rob_x(){return RState(S);}
    RState rob_x_old(){return RState(S_old);}
    bool pos_init;

    std::vector<ObjectDataPtr> adv_erase_obj;
    std::map<SegmentDataExtPtr, ObjectDataPtr> seg_ext_new_obj;
    std::vector<SegmentDataPtr> adv_no_innv_seg;
    void advance(InputData& input, bool advance);
    //advance-erase-object list
    //advance-segext-new-obj    ....make it a map...the ones that are not in here get erased


private:

    void predict_rob(RState  rob_f0, KInp u, cv::Mat& Gt_R, cv::Mat& Q);
    void predict_obj(KInp u, cv::Mat &Gt, cv::Mat& Q);
    void predict_p_cloud(SegmentDataPtrVectorPtr &input, RState  rob_f0, KInp u);

    std::map<ObjectDataPtr, ObjMat> Oi_old;
    cv::Mat S_R_bar_old;
    cv::Mat S_old;
    cv::Mat P_old;

    SegmentDataPtrVectorPtr    seg_init_plus;

    cv::Mat Gt_Oi(double dt);
    cv::Mat Q_Oi (double dt);

    bool add_obj(ObjectDataPtr seg, KObjZ kObjZ);
    bool rmv_obj(ObjectDataPtr seg);
    cv::Mat Fxi(ObjectDataPtr seg);
    void update_sub_mat();

    static const int rob_param   = 3;
    static const int obj_param   = 9;
    static const int input_param = 3;
    static const int z_param     = 3;
    double input_noise[5];
    double obj_noise_x;
    double obj_noise_y;
    double obj_noise_phi;

    double v_static;
    double w_static;


    cv::Mat P_RO;
    cv::Mat P_OR;
    cv::Mat P_OO;
};


#endif // KALMAN_H
