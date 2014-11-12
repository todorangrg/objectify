#ifndef KALMAN_H
#define KALMAN_H
#include "opencv/cv.h"
#include "utils/base_classes.h"

class NeighData;

class OiState{
public:
    double xx; double xy; double xphi;
    double vx; double vy; double vphi;
    double ax; double ay; double aphi;
    OiState(cv::Mat _S_O);
    OiState(){}
};
class RState{
public:
    double xx; double xy; double xphi;
    RState(cv::Mat _S);
    RState(){}
};


class ObjMat{
public:
    int i_min;
    cv::Mat S_O;
    cv::Mat P_OO;
};


class KalmanSLDM{
public:
    void run(SensorData& sensor, std::map <SegmentDataExtPtr, std::vector<NeighData> >& neigh_data_to, std::map <SegmentDataExtPtr, std::vector<NeighData> >& neigh_data_tn);


    void init(State rob_x);
    bool add_obj(ObjectDataPtr seg, KObjZ kObjZ);
    bool rmv_obj(ObjectDataPtr seg);
    void prediction(KControl u);
    void init_Oi(ObjectDataPtr obj, xy obj_com_bar_f1);
    void update_Oi(ObjectDataPtr seg, KObjZ kObjZ);
    void update_Oi_with_Oj(ObjectDataPtr seg, ObjectDataPtr seg_obs);


    std::map<ObjectDataPtr, ObjMat> Oi;
    cv::Mat S_R_bar;
    cv::Mat S;
    cv::Mat P;






    bool pos_init;

    std::vector<ObjectDataPtr> adv_erase_obj;
    std::map<SegmentDataExtPtr, ObjectDataPtr> seg_ext_new_obj;
    std::vector<SegmentDataPtr> adv_no_innv_seg;
    void advance(SensorData& sensor,bool advance);
    //advance-erase-object list
    //advance-segext-new-obj    ....make it a map...the ones that are not in here get erased


private:

    std::map<ObjectDataPtr, ObjMat> Oi_old;
    cv::Mat S_R_bar_old;
    cv::Mat S_old;
    cv::Mat P_old;

    cv::Mat Gt_Oi(double dt);
    cv::Mat Q_Oi (double dt);

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

    //cv::Mat M;

//    cv::Mat Gt_R;
//    cv::Mat Gt_O;
//    cv::Mat Gt;

//    cv::Mat Vt_R;
    //cv::Mat Vt_O;
    //cv::Mat Vt;

//    cv::Mat Ht_low;

    //cv::Mat S_R;

    cv::Mat P_RO;
    cv::Mat P_OR;
    cv::Mat P_OO;

    //cv::Mat M_R;
    //cv::Mat M_O;
};


#endif // KALMAN_H
