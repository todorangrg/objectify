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


#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;

///------------------------------------------------------------------------------------------------------------------------------------------------///

KalmanSLDM::KalmanSLDM(RecfgParam& _param, SensorTf& _tf_sns , rosbag::Bag &_bag) :
    plotw("World View", _param, _tf_sns, *this),
    bag(_bag),
    tf_sns                 (_tf_sns),

    alfa_ini_obj_pow_dt    (_param.kalman_alfa_ini_obj_pow_dt),
    alfa_pre_obj_xy_min    (_param.kalman_alfa_pre_obj_xy_min),
    alfa_pre_obj_phi       (_param.kalman_alfa_pre_obj_phi),
    alfa_dsc_surface       (_param.kalman_alfa_dsc_obj_surface),
    discard_old_seg_perc   (_param.kalman_discard_old_seg_perc),
    no_upd_vel_hard0       (_param.kalman_no_upd_vel_hard0),

    alfa_pre_rob_v_base    (_param.kalman_alfa_pre_rob_v_base),
    alfa_pre_rob_w_base    (_param.kalman_alfa_pre_rob_w_base),
    alfa_upd_rob_vv        (_param.kalman_alfa_upd_rob_vv),
    alfa_upd_rob_ww        (_param.kalman_alfa_upd_rob_ww),

    adaptive_resid_min     (_param.kalman_adaptive_resid_min),
    adaptive_scale_bound   (_param.kalman_adaptive_scale_bound),
    adaptive_noise_scale   (_param.kalman_adaptive_noise_scale),

    adpt_obj_resid_scale   (_param.kalman_adpt_obj_resid_scale),

    dynamic_obj            (_param.kalman_dynamic_obj),

    pos_init             (false),
    bag_file_n("LocalizationTest"){

    seg_init     = SegmentDataPtrVectorPtr(new SegmentDataPtrVector);
    seg_init_old = SegmentDataPtrVectorPtr(new SegmentDataPtrVector);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init(RState rob_x){
    rob_odom = rob_x;
    pos_init = true;
    residuals.clear();
    S.release(); S_bar.release(); P.release();;
    Oi.clear();

    S_bar.push_back(rob_x.xx); S_bar.push_back(rob_x.xy); S_bar.push_back(rob_x.xphi); S_bar.push_back(0.); S_bar.push_back(0.); S_bar.push_back(0.); S_bar.push_back(0.);
    S    .push_back(rob_x.xx); S    .push_back(rob_x.xy); S    .push_back(rob_x.xphi); S    .push_back(0.); S    .push_back(0.); S    .push_back(0.); S    .push_back(0.);
    P = Mat::zeros(rob_param, rob_param, CV_64F);

    S_bar_old.push_back(rob_x.xx); S_bar_old.push_back(rob_x.xy); S_bar_old.push_back(rob_x.xphi); S_bar_old.push_back(0.); S_bar_old.push_back(0.); S_bar_old.push_back(0.); S_bar_old.push_back(0.);
    S_old    .push_back(rob_x.xx); S_old    .push_back(rob_x.xy); S_old    .push_back(rob_x.xphi); S_old    .push_back(0.); S_old    .push_back(0.); S_old    .push_back(0.); S_old    .push_back(0.);
    P_old = Mat::zeros(rob_param, rob_param, CV_64F);

    if(dynamic_obj){
        Mat Q_rob_init(rob_param,rob_param, CV_64F, 0.);//HERE INIT POS UNCERTAINTY ALSO!!!
        Q_rob_init.row(3).col(3) = 10 * alfa_pre_rob_v_base;
        Q_rob_init.row(4).col(4) = 10 * alfa_pre_rob_w_base;
        Q_rob_init.copyTo(P.rowRange(0,rob_param).colRange(0,rob_param));
    }
    P.copyTo(P_old);
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
    //cout<<"M="<<endl<<" "<<M<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool KalmanSLDM::add_obj(ObjectDataPtr seg, KObjZ kObjZ){
    //cout<<"adding object with id"<<endl;
    if(Oi.count(seg) > 0){
        //cout<<"warning, adding an allready inserted object"<<endl;
        return false;
    }
    RState  rob_bar_f0(S);
    OiState obj_zhat_f1;
    obj_zhat_f1.xx = kObjZ.pos.x; obj_zhat_f1.xy = kObjZ.pos.y; obj_zhat_f1.xphi = kObjZ.phi;

    //adding the object state parameters
    S.push_back(rob_bar_f0.xx +
                cos(rob_bar_f0.xphi) * tf_sns.getXY().x -
                sin(rob_bar_f0.xphi) * tf_sns.getXY().y +
                cos(tf_sns.getPhi()) * (   obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi)) +
                sin(tf_sns.getPhi()) * ( - obj_zhat_f1.xy * cos(rob_bar_f0.xphi) - obj_zhat_f1.xx * sin(rob_bar_f0.xphi)));
    S.push_back(rob_bar_f0.xy +
                cos(rob_bar_f0.xphi) * tf_sns.getXY().y +
                sin(rob_bar_f0.xphi) * tf_sns.getXY().x +
                cos(tf_sns.getPhi()) * (   obj_zhat_f1.xy * cos(rob_bar_f0.xphi) + obj_zhat_f1.xx * sin(rob_bar_f0.xphi)) +
                sin(tf_sns.getPhi()) * (   obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi)));
    S.push_back(rob_bar_f0.xphi + obj_zhat_f1.xphi + tf_sns.getPhi());
    S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 );

    hconcat(P, Mat(P.rows, obj_param, CV_64F, 0.), P);
    P.resize(P.cols,0.);//resize-ing P
    update_sub_mat();   //re-binding sub-matrices masks

    Oi[seg].i_min = P.rows - obj_param;//mapping object specific information
    Oi[seg].S_O   = S.rowRange(S.rows - obj_param, S.rows);
    Mat P_ORi = P_OR.rowRange(P_OR.rows - obj_param, P_OR.rows);
    Mat P_ROi = P_RO.colRange(P_RO.cols - obj_param, P_RO.cols);
    Oi[seg].P_OO  = P.rowRange(P.rows - obj_param, P.rows).colRange(P.cols - obj_param, P.cols);

    ///OBJECT COVARIANCE INITIALIZATION----
    Mat Gt_hinv_R (obj_param, rob_param, CV_64F, 0.);

    Gt_hinv_R.row(0).col(0) = 1.;
    Gt_hinv_R.row(1).col(1) = 1.;
    Gt_hinv_R.row(2).col(2) = 1.;

    Gt_hinv_R.row(0).col(2) = - cos(rob_bar_f0.xphi) * (tf_sns.getXY().y + obj_zhat_f1.xy * cos(tf_sns.getPhi()) + obj_zhat_f1.xx * sin(tf_sns.getPhi()))
                              - sin(rob_bar_f0.xphi) * (tf_sns.getXY().x + obj_zhat_f1.xx * cos(tf_sns.getPhi()) - obj_zhat_f1.xy * sin(tf_sns.getPhi()));
    Gt_hinv_R.row(1).col(2) =   cos(rob_bar_f0.xphi) * (tf_sns.getXY().x + obj_zhat_f1.xx * cos(tf_sns.getPhi()) - obj_zhat_f1.xy * sin(tf_sns.getPhi()))
                              - sin(rob_bar_f0.xphi) * (tf_sns.getXY().y + obj_zhat_f1.xy * cos(tf_sns.getPhi()) + obj_zhat_f1.xx * sin(tf_sns.getPhi()));

    Mat P_OO_V(9, 9, CV_64F, 0.);


    if(dynamic_obj){
        Q_Oi(alfa_pre_obj_xy_min, alfa_pre_obj_phi, alfa_ini_obj_pow_dt).copyTo(P_OO_V);
        Mat(Gt_hinv_R * P.rowRange(0, rob_param).colRange(0, rob_param) * Gt_hinv_R.t()  + P_OO_V/*Gt_hinv_Z * Mat(kObjZ.Q)  * Gt_hinv_Z.t()*/).copyTo(Oi[seg].P_OO);
    }
    else{
        Mat Gt_hinv_Z (obj_param, z_param, CV_64F, 0.);
        Gt_hinv_Z.row(0).col(0) = 1.;//-sin(rob_bar_f0.xphi)*sin(tf_sns.getPhi())+cos(tf_sns.getPhi())*cos(rob_bar_f0.xphi);
//        Gt_hinv_Z.row(0).col(1) = -cos(rob_bar_f0.xphi)*sin(tf_sns.getPhi())-cos(tf_sns.getPhi())*sin(rob_bar_f0.xphi);
//        Gt_hinv_Z.row(1).col(0) =  cos(rob_bar_f0.xphi)*sin(tf_sns.getPhi())+cos(tf_sns.getPhi())*sin(rob_bar_f0.xphi);
        Gt_hinv_Z.row(1).col(1) = 1.;//-sin(rob_bar_f0.xphi)*sin(tf_sns.getPhi())+cos(tf_sns.getPhi())*cos(rob_bar_f0.xphi);
        Gt_hinv_Z.row(2).col(2) = 1.;
        kObjZ.Q = cv::Matx33d(100., 0., 0.,
                0., 100., 0.,
                0., 0., 100.);
        Mat(Gt_hinv_R * P.rowRange(0, rob_param).colRange(0, rob_param) * Gt_hinv_R.t()  + Gt_hinv_Z * Mat(kObjZ.Q)   * Gt_hinv_Z.t()).copyTo(Oi[seg].P_OO);
    }



    Mat(Gt_hinv_R * P.rowRange(0, rob_param).colRange(0, P.cols - obj_param)).copyTo(P_ORi);
    Mat(P_ORi.t()).copyTo(P_ROi);
    ///OBJECT COVARIANCE INITIALIZATION----

    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool KalmanSLDM::rmv_obj(ObjectDataPtr seg){
    //cout<<"removing object with id"<<test_id<<endl;
    if(Oi.count(seg) == 0){
        //cout<<"attempting to remove inexisting key"<<endl;
        return false;
    }
    int i_min = Oi[seg].i_min;
    Mat temp = S.rowRange(0, i_min).clone();
    temp.push_back(S.rowRange(i_min + obj_param, S.rows));
    S = temp;//removing the object state parameters
    temp.release();

    temp = P.rowRange(0,i_min).clone();
    temp.push_back(P.rowRange(i_min + obj_param, P.rows));
    P    = temp;
    temp = P.colRange(0,i_min).clone();
    if(P.cols - i_min - obj_param > 0){
        hconcat(temp, P.colRange(i_min + obj_param, P.cols), temp);
    }
    P = temp;//resize-ing p

    Oi.erase(seg);//erasing the object key
    for(map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin();oi != Oi.end(); oi++){
        if(oi->second.i_min > i_min){
            oi->second.i_min-= obj_param;
        }
    }
    update_sub_mat();

    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::update_sub_mat(){
    if( P.rows - rob_param > 0 ){
        P_OR = P.rowRange(rob_param, P.rows   ).colRange(0        , rob_param);
        P_RO = P.rowRange(0        , rob_param).colRange(rob_param, P.cols   );
        P_OO = P.rowRange(rob_param, P.rows   ).colRange(rob_param, P.cols   );
    }
    else{
        P_OR.release(); P_RO.release(); P_OO.release();
    }
    for(map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin();oi != Oi.end(); oi++){
        int i_min = oi->second.i_min;
        oi->second.S_O   = S.rowRange(i_min, i_min + obj_param);
        oi->second.P_OO  = P.rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

cv::Mat KalmanSLDM::Fxi(ObjectDataPtr seg){
    cv::Mat Fx(rob_param + obj_param, rob_param + obj_param * Oi.size(), CV_64F, 0.);
    for(int i = 0;i < rob_param; i++){ Fx.row(i).col(i)                             = 1.; }
    for(int i = 0;i < obj_param; i++){ Fx.row(rob_param + i).col(Oi[seg].i_min + i) = 1.; }
    return Fx;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

int KalmanSLDM::assign_unique_obj_id(){
    const int max_no_obj = 50;
    bool ids[max_no_obj];
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        ids[oi->first->id] = true;
    }
    for(int i=0; i < max_no_obj; i++){
        if(ids[i] == false){
            return i;
        }
    }
    return -1;
}
