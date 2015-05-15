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


#ifndef KALMAN_H
#define KALMAN_H
#include "opencv/cv.h"
#include "utils/base_classes.h"
#include "visual/plot_world.h"

class NeighData;
class Segmentation;

class update_info{
public:
    ObjectDataPtr obj;
    cv::Mat h_diff;
    cv::Mat Q;
//    cv::Mat Ht_low;
    xy avg_com;
    KObjZ kObjZ;
    cv::Mat h_hat;
    update_info(ObjectDataPtr _obj, cv::Mat _h_diff, cv::Mat _Q, /*cv::Mat _Ht_low,*/ xy _avg_com, KObjZ _kObjZ, cv::Mat _h_hat){
        obj = _obj;
        _h_diff.copyTo(h_diff);
        _Q.copyTo(Q);
//        _Ht_low.copyTo(Ht_low);
        _h_hat.copyTo(h_hat);
        avg_com = _avg_com;
        kObjZ = _kObjZ;
    }
    ~update_info(){}
};


class ObjMat{
public:
    int i_min;
    cv::Mat S_O;
    cv::Mat P_OO;
    cv::Mat S_O_old;
    KObjZ   last_upd_info;
    bool symmetry_flag;
    double symmetry_vec_x;
    double symmetry_vec_y;//TODO: messy stuff

    boost::shared_ptr<std::vector<double> > residuals;
    ObjMat(){residuals = boost::shared_ptr<std::vector<double> >(new std::vector<double> );}
    ~ObjMat(){}
};

class KalmanSLDM{
public:
    PlotWorld plotw;


    //mainly for tangent bug
    template <class SegData>
    void predict_p_cloud(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &input, RState  rob_f0, double dt);
    RState predict_rob_pos(RState  rob_f0, double dt);
    xy goal;



    bool pos_init;

    rosbag::Bag &bag;
    std::string bag_file_n;
    double&   no_upd_vel_hard0;

    SegmentDataPtrVectorPtr    seg_init;
    SegmentDataExtPtrVectorPtr seg_ext;
    ros::Time                  time_stamp;
    std::map<ObjectDataPtr, ObjMat> Oi;
    cv::Mat S;
    cv::Mat S_bar;
    cv::Mat P;

    RState rob_real;
    RState rob_odom;

    void init      (RState rob_x);
    void prediction(SegmentDataPtrVectorPtr &input, KInp &u);
    void advance   (InputData & input, bool advance);
    void run       (InputData & input,
                    std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >  & neigh_data_oe,
                    std::map <SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_ne,
                    std::map <SegmentDataPtr   , std::vector<NeighDataInit> > & neigh_data_oi,
                    std::map <SegmentDataPtr   , std::vector<NeighDataInit> > & neigh_data_ni, Segmentation & segmentation);

    RState rob_x_now(){return RState(S);}
    RState rob_x_bar(){return RState(S_bar);}
    RState rob_x_old(){return RState(S_old);}

    //Constructors & Destructors
    KalmanSLDM(RecfgParam& _param, SensorTf& _tf_sns, rosbag::Bag &bag);
    ~KalmanSLDM(){}
private:
    static const int rob_param   = 7;
    static const int obj_param   = 9;
    static const int input_param = 2;
    static const int z_param     = 3;
    static const int z_rob_param = 2;

    //adaptive filtering--
    std::vector<double> residuals;
    std::vector<double> residuals_old;
    int epoch_no;
    std::vector<update_info> updates;
    void apply_obj_updates(std::vector<update_info>::iterator upd);
    void store_w_residuals(bool second_time);
    bool adaptive_noise_rob();
    void adaptive_noise_obj(double dt);
    double & adpt_obj_resid_scale;
    double & adaptive_resid_min;
    double & adaptive_scale_bound;
    double & adaptive_noise_scale;
    //--adaptive filtering

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
    void predict_rob    (RState  rob_f0, double dt, cv::Mat& Gt_R, cv::Mat& Q);
    cv::Mat Vt_R;
    cv::Mat Vt_a;
    void predict_obj    (KInp u, cv::Mat &Gt, cv::Mat& Q);

    cv::Mat Gt_Oi(double dt);
    cv::Mat Q_Oi (double _obj_alfa_xy, double _obj_alpha_phi, double dt);
    // ---kalman_prediction

    //kalman_update ---
    void init_Oi  (ObjectDataPtr obj, xy obj_com_bar_f1, double dt);
    void update_Oi(ObjectDataPtr seg, KObjZ kObjZ, xy avg_com, double post_upd);
    void update_rob(KInp u);
    // ---kalman_update

    //kalman_update ---
    void extract_common_pairs   (std::vector<ObjectDataPtr>                               &    o_comm,
                                 std::vector<CorrInput>                                   & list_comm,
                                 std::map<SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_oe,
                                 std::map<SegmentDataExtPtr, std::vector<NeighDataExt > > & neigh_data_ne);
    void propagate_no_update_obj(std::map <SegmentDataPtr  , std::vector<NeighDataInit> > & neigh_data_oi,
                                 std::map <SegmentDataPtr  , std::vector<NeighDataInit> > & neigh_data_ni, double _dt, Segmentation & segmentation);
    bool compute_avg_miu_sigma(std::vector<CorrInput> & list_comm, KObjZ & avg, xy &avg_com);
    void propag_extr_p_clouds (std::vector<CorrInput> & list_comm, std::map<ObjectDataPtr  , ObjMat>::iterator                        oi);
    void add_new_obj          (SegmentDataPtrVectorPtr & input   , std::map <SegmentDataPtr   , std::vector<NeighDataInit> >& neigh_data_ni
                                                                 , std::map <SegmentDataExtPtr, std::vector<NeighDataExt>  >& neigh_data_ne);
    void remove_lost_obj();
    // ---kalman_update

    double&   alfa_ini_obj_pow_dt;

    double&   alfa_pre_rob_v_base;
    double&   alfa_pre_rob_w_base;

    double&   alfa_upd_rob_vv;
    double&   alfa_upd_rob_ww;

    double&   alfa_pre_obj_xy_min;
    double&   alfa_pre_obj_phi;

    double&   alfa_dsc_surface;

    double&   discard_old_seg_perc;

    double&   dynamic_obj;


    SensorTf& tf_sns;
};

extern template void KalmanSLDM::predict_p_cloud(SegmentDataPtrVectorPtr    &input, RState  rob_f0, double dt);
extern template void KalmanSLDM::predict_p_cloud(SegmentDataExtPtrVectorPtr &input, RState  rob_f0, double dt);


#endif // KALMAN_H
