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

void KalmanSLDM::update_rob(KInp u){
    RState rob_bar(S);
    Mat h_bar(input_param, 1, CV_64F, 0.0);
    h_bar.row(0) = S.at<double>(3);
    h_bar.row(1) = S.at<double>(4);
    Mat h_hat(input_param, 1, CV_64F, 0.0);
    h_hat.row(0) = u.v;
    h_hat.row(1) = u.w;

    Mat Ht(input_param, rob_param + Oi.size() * obj_param, CV_64F, 0.);
    Ht.row(0).col(3) = 1.;
    Ht.row(1).col(4) = 1.;


    Mat M = Mat::zeros(input_param , input_param, CV_64F);
    M.row(0).col(0) = sqr(alfa_upd_rob_vv * 100 /** rob_bar.alin*/) + 0.00001;
    M.row(1).col(1) = sqr(alfa_upd_rob_ww * 100 /** rob_bar.aang*/) + 0.00001;

    Mat Kt = P * Ht.t() * ( Ht * P * Ht.t() + M ).inv(DECOMP_SVD);      ///KALMAN GAIN
    Mat(S + Kt * ( h_hat - h_bar )).copyTo(S);                          ///STATE UPDATE
    Mat((cv::Mat::eye(P.rows, P.cols, CV_64F) - Kt * Ht) * P).copyTo(P);///COVARIANCE UPDATE

    //cout<<"h_diff_f1="<<endl<<" "<<Mat(h_hat - h_bar)        <<endl<<endl;
    //std::cout<<"v="<<S.at<double>(3)<<", w="<<S.at<double>(4)<<std::endl;
    //cout<<"Kt="       <<endl<<" "<<Kt                              <<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init_Oi(ObjectDataPtr obj, xy obj_com_bar_f1, double dt){
    RState rob_bar_f0(S_bar);
    if(Oi.count(obj) == 0){ return; }
    int i_min = Oi[obj].i_min;

    //TODO: take symmetry stuff into account here (predict with velocity stuff along the symmetry direction) (now messy done)
    //TODO: STILL BUGGY!!!!
    
    S.row(i_min + 0) = rob_bar_f0.xx +
                       cos(rob_bar_f0.xphi) * tf_sns.getXY().x -
                       sin(rob_bar_f0.xphi) * tf_sns.getXY().y +
                       cos(tf_sns.getPhi()) * (   obj_com_bar_f1.x * cos(rob_bar_f0.xphi) - obj_com_bar_f1.y * sin(rob_bar_f0.xphi)) +
                       sin(tf_sns.getPhi()) * ( - obj_com_bar_f1.y * cos(rob_bar_f0.xphi) - obj_com_bar_f1.x * sin(rob_bar_f0.xphi));
    S.row(i_min + 1) = rob_bar_f0.xy +
                       cos(rob_bar_f0.xphi) * tf_sns.getXY().y +
                       sin(rob_bar_f0.xphi) * tf_sns.getXY().x +
                       cos(tf_sns.getPhi()) * (   obj_com_bar_f1.y * cos(rob_bar_f0.xphi) + obj_com_bar_f1.x * sin(rob_bar_f0.xphi)) +
                       sin(tf_sns.getPhi()) * (   obj_com_bar_f1.x * cos(rob_bar_f0.xphi) - obj_com_bar_f1.y * sin(rob_bar_f0.xphi));
    S.row(i_min + 2) =  (S.row(i_min + 5) * dt + S.row(i_min + 8) * sqr(dt) / 2.0) + rob_bar_f0.xphi ;
    
    if(Oi[obj].symmetry_flag){
	OiState obj_f0;
	obj_f0.init(Oi[obj].S_O);
	
	obj_f0.vx = obj_f0.vx * Oi[obj].symmetry_vec_x;
	obj_f0.vy = obj_f0.vy * Oi[obj].symmetry_vec_y;
	obj_f0.ax = obj_f0.ax * Oi[obj].symmetry_vec_x;
	obj_f0.ay = obj_f0.ay * Oi[obj].symmetry_vec_y;
	
	S.row(i_min + 0) += obj_f0.vx * dt + obj_f0.ax * sqr(dt) / 2.0;
	S.row(i_min + 1) += obj_f0.vy * dt + obj_f0.ay * sqr(dt) / 2.0;
    }
    Mat(S.rowRange(i_min, i_min + obj_param)).copyTo(Oi[obj].S_O_old);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::update_Oi(ObjectDataPtr seg, KObjZ kObjZ, xy avg_com, double post_upd){
    if(add_obj(seg, kObjZ)){ return; }
    //cout<<"innovation:"<<endl;

    RState  rob_bar_f0(S/*_bar*/);
    OiState obj_bar_f0;
    obj_bar_f0.init(Oi[seg].S_O_old);

    ///PREDICTED OBSERVATION----
    Mat h_bar_f1(z_param, 1, CV_64F, 0.0);

    h_bar_f1.row(0) = cos(tf_sns.getPhi()) * ( - tf_sns.getXY().x + (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) +
                                                                    (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi)) +
                      sin(tf_sns.getPhi()) * ( - tf_sns.getXY().y + (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi));
    h_bar_f1.row(1) = cos(tf_sns.getPhi()) * ( - tf_sns.getXY().y + (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi)) +
                      sin(tf_sns.getPhi()) * ( + tf_sns.getXY().x - (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi));
    h_bar_f1.row(2) =  obj_bar_f0.xphi - rob_bar_f0.xphi;
    ///----PREDICTED OBSERVATION

//    ///PREDICTED OBSERVATION JACOBIAN
//    Mat Ht_low(Mat(z_param, rob_param + obj_param, CV_64F, 0.));

//    Ht_low.row(0).col(0) = - cos(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
//    Ht_low.row(0).col(1) = - sin(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) - cos(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
//    Ht_low.row(1).col(0) =   cos(rob_bar_f0.xphi) * sin(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * cos(tf_sns.getPhi());
//    Ht_low.row(1).col(1) = - cos(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
//    Ht_low.row(2).col(2) = - 1.;

//    Ht_low.row(0).col(2) =    cos(tf_sns.getPhi()) * (   (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) - (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi))
//                            - sin(tf_sns.getPhi()) * (   (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) + (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi));
//    Ht_low.row(1).col(2) =    cos(tf_sns.getPhi()) * ( - (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) - (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi)) +
//                              sin(tf_sns.getPhi()) * ( - (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) + (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi));

//    Ht_low.row(0).col(rob_param + 0) = + cos(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - sin(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
//    Ht_low.row(0).col(rob_param + 1) =   sin(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) + cos(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
//    Ht_low.row(1).col(rob_param + 0) = - sin(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - cos(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
//    Ht_low.row(1).col(rob_param + 1) = + cos(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - sin(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
//    Ht_low.row(2).col(rob_param + 2) =  1.;
//    ///----PREDICTED OBSERVATION JACOBIAN

    Mat Q(kObjZ.Q);
    Mat h_hat_f1(z_param, 1, CV_64F, 0.0); h_hat_f1.row(0) = kObjZ.pos.x; h_hat_f1.row(1) = kObjZ.pos.y; h_hat_f1.row(2) = kObjZ.phi; ///OBSERVATION

    Mat h_diff = h_hat_f1 - h_bar_f1;
    double obj_angle = h_diff.at<double>(2); h_diff.row(2) = normalizeAngle(obj_angle);

    updates.push_back(update_info(seg, h_diff, Q, /*Ht_low,*/ avg_com, kObjZ, h_hat_f1));
    //cout<<"----------------"<<endl;
    //cout<<"angle_hat"<<h_hat_f1.at<double>(2)<<endl;
    //cout<<"angle_bar"<<h_bar_f1.at<double>(2)<<endl;
    //cout<<"angle_diff"<<h_diff.at<double>(2)<<endl;

    //cout<<"h_hat_f1=" <<endl<<" "<<h_hat_f1                        <<endl<<endl;
    //cout<<"h_bar_f1=" <<endl<<" "<<h_bar_f1                        <<endl<<endl;
    //cout<<"h_diff_f1="<<endl<<" "<<Mat(h_hat_f1 - h_bar_f1)        <<endl<<endl;
    //cout<<"Kt="       <<endl<<" "<<Kt                              <<endl<<endl;
    //cout<<"Kt*dh"     <<endl<<" "<<Mat(Kt * (h_hat_f1 - h_bar_f1 ))<<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
}



void KalmanSLDM::apply_obj_updates(std::vector<update_info>::iterator upd){

    RState  rob_bar_f0(S);
    OiState obj_bar_f0(Oi[upd->obj].S_O);

    ///PREDICTED OBSERVATION----
    Mat h_bar_f1(z_param, 1, CV_64F, 0.0);

    h_bar_f1.row(0) = cos(tf_sns.getPhi()) * ( - tf_sns.getXY().x + (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) +
                                                                    (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi)) +
                      sin(tf_sns.getPhi()) * ( - tf_sns.getXY().y + (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi));
    h_bar_f1.row(1) = cos(tf_sns.getPhi()) * ( - tf_sns.getXY().y + (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi)) +
                      sin(tf_sns.getPhi()) * ( + tf_sns.getXY().x - (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) -
                                                                    (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi));
    h_bar_f1.row(2) =  obj_bar_f0.xphi - rob_bar_f0.xphi;

    Mat Ht_low(Mat(z_param, rob_param + obj_param, CV_64F, 0.));

    Ht_low.row(0).col(0) = - cos(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
    Ht_low.row(0).col(1) = - sin(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) - cos(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
    Ht_low.row(1).col(0) =   cos(rob_bar_f0.xphi) * sin(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * cos(tf_sns.getPhi());
    Ht_low.row(1).col(1) = - cos(rob_bar_f0.xphi) * cos(tf_sns.getPhi()) + sin(rob_bar_f0.xphi) * sin(tf_sns.getPhi());
    Ht_low.row(2).col(2) = - 1.;

    Ht_low.row(0).col(2) =    cos(tf_sns.getPhi()) * (   (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) - (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi))
                            - sin(tf_sns.getPhi()) * (   (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) + (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi));
    Ht_low.row(1).col(2) =    cos(tf_sns.getPhi()) * ( - (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) - (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi)) +
                              sin(tf_sns.getPhi()) * ( - (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) + (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi));

    Ht_low.row(0).col(rob_param + 0) = + cos(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - sin(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
    Ht_low.row(0).col(rob_param + 1) =   sin(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) + cos(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
    Ht_low.row(1).col(rob_param + 0) = - sin(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - cos(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
    Ht_low.row(1).col(rob_param + 1) = + cos(tf_sns.getPhi()) * cos(rob_bar_f0.xphi) - sin(tf_sns.getPhi()) * sin(rob_bar_f0.xphi);
    Ht_low.row(2).col(rob_param + 2) =  1.;
    ///----PREDICTED OBSERVATION


    Mat Q(upd->kObjZ.Q);
    Mat h_hat_f1(z_param, 1, CV_64F, 0.0); h_hat_f1.row(0) = upd->kObjZ.pos.x; h_hat_f1.row(1) = upd->kObjZ.pos.y; h_hat_f1.row(2) = upd->kObjZ.phi; ///OBSERVATION

    Mat h_diff = h_hat_f1 - h_bar_f1;
    double obj_angle = h_diff.at<double>(2); h_diff.row(2) = /*0.;//*/normalizeAngle(obj_angle);

    Mat Ht = /*upd->*/Ht_low * Fxi(upd->obj);
    Mat Kt = P * Ht.t() * ( Ht * P * Ht.t() + /*upd->*/Q ).inv(DECOMP_SVD);///KALMAN GAIN
    Mat(S + Kt * ( /*upd->*/h_diff )).copyTo(S);                    ///STATE UPDATE
    Mat((cv::Mat::eye(P.rows, P.cols, CV_64F) - Kt * Ht) * P).copyTo(P);///COVARIANCE UPDATE
    
    Oi[upd->obj].last_upd_info = upd->kObjZ;
    //std::cout<<"x= "<<S.at<double>(0)<<", y= "<<S.at<double>(1)<<", a= "<<S.at<double>(2)<<std::endl;
    //std::cout<<"resid obj : x= "<<h_diff.at<double>(0)<<", y= "<<h_diff.at<double>(1)<<", a= "<<h_diff.at<double>(2)<<std::endl;
}

bool KalmanSLDM::adaptive_noise_rob(){

    if(residuals.size() < 2){ return false; }
    double a = residuals.back();
    double b = 0;
    //std::cout<<"old residuals = ";///

    for(std::vector<double>::iterator res = residuals.begin(); res != --residuals.end(); res++){
        b += *res; //std::cout<<*res<<", ";///
    }
    //std::cout<<endl; std::cout<<"new residual = "<<a<<std::endl;///
    b /= double(residuals.size() - 1);

    double s = a / b; if(b == 0){ s = 0; }
    if     (s <  adaptive_scale_bound){s = 1.0; }
    //std::cout<<"s = "<<s<<std::endl;///

    s = s - 1.0;
    Mat M = Mat::zeros(rob_param , rob_param, CV_64F);
    M.row(0).col(0) = alfa_pre_rob_v_base * adaptive_noise_scale * s;
    M.row(1).col(1) = alfa_pre_rob_v_base * adaptive_noise_scale * s;
    M.row(2).col(2) = alfa_pre_rob_w_base * adaptive_noise_scale * s;
    (Mat(P.rowRange(0, rob_param).colRange(0, rob_param) + M)).copyTo(P.rowRange(0, rob_param).colRange(0, rob_param));

//    Mat M = Mat::zeros(input_param , input_param, CV_64F);
//    M.row(0).col(0) = /*sqr(*/alfa_pre_rob_v_base * adaptive_noise_scale * s/** dt * rob_f0.alin)*//* + 0.0001*/;// MULTIPLY WITH DT TODO
//    M.row(1).col(1) = /*sqr(*/alfa_pre_rob_w_base * adaptive_noise_scale * s/** dt * rob_f0.aang)*//* + 0.0001*/;
//    (Mat(P.rowRange(0, rob_param).colRange(0, rob_param) + Vt_a * M * Vt_a.t())).copyTo(P.rowRange(0, rob_param).colRange(0, rob_param));
    if(s == 0){
        return false;
    }
    return true;
}


void KalmanSLDM::adaptive_noise_obj(double dt){
    for(std::vector<update_info>::iterator upd = updates.begin(); upd != updates.end(); upd++){
        if(Oi[upd->obj].residuals->size() < 2){ continue; }
        int i_min = Oi[upd->obj].i_min;
        double s = Oi[upd->obj].residuals->back() * adpt_obj_resid_scale;//std::cout<<"obj_s= "<<s<<std::endl;
        Mat(P .rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param) +
            Q_Oi(s * alfa_pre_obj_xy_min,s * alfa_pre_obj_phi, dt))
                .copyTo(P .rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param));
    }
}

void KalmanSLDM::store_w_residuals(bool second_time){//weight on number of updates
    double residual = 0.;
    for(std::vector<update_info>::iterator upd = updates.begin(); upd != updates.end(); upd++){
        if(Oi.count(upd->obj) == 0){continue; }
        Mat cov_obj_v     = P.rowRange(Oi[upd->obj].i_min + 3, Oi[upd->obj].i_min + 6).colRange(Oi[upd->obj].i_min + 3, Oi[upd->obj].i_min + 6);
        upd->h_diff.row(2) = 0.;
        Mat h_diff_weight = upd->h_diff.t() * cov_obj_v.inv(DECOMP_SVD) * upd->h_diff;
        residual += (Mat(h_diff_weight)).at<double>(0);
        if(second_time){
            Oi[upd->obj].residuals->erase(--Oi[upd->obj].residuals->end());
        }
        Oi[upd->obj].residuals->push_back((Mat(h_diff_weight)).at<double>(0));

        while(Oi[upd->obj].residuals->size() > epoch_no + 1){
            Oi[upd->obj].residuals->erase(Oi[upd->obj].residuals->begin());
        }
    }

    if( updates.size() > 0){
	residual /= (double)updates.size();
	if(second_time){
	    residuals.erase(--residuals.end());
	}
    }
    residuals.push_back(fmax(residual, adaptive_resid_min));
    while(residuals.size() > epoch_no + 1){
	residuals.erase(residuals.begin());
    }
//    if(!second_time){
//        if( updates.size() > 0){ residual /= (double)updates.size(); }
//        residuals.push_back(fmax(residual, adaptive_resid_min));
//        while(residuals.size() > epoch_no + 1){
//            residuals.erase(residuals.begin());
//        }
//    }
}
