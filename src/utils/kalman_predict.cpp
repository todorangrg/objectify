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

void KalmanSLDM::prediction(SegmentDataPtrVectorPtr &input, KInp &u){
    //cout<<"prediction:"<<endl;
    if(!pos_init){return; }
    if(u.v != u.v){ u.v = 0; }//correcting NaN stuff
    if(u.w != u.w){ u.w = 0; }

    Mat Gt (Mat::eye(P.rows, P.cols, CV_64F));
    Mat Gt_R = Gt.rowRange(0, rob_param).colRange(0, rob_param);

    Mat Q(P.rows,P.cols,CV_64F, 0.);
    RState  rob_f0(S);

    predict_p_cloud(input, rob_f0, u.dt);
    predict_obj(u,Gt, Q);

    predict_rob(rob_f0, u.dt, Gt_R, Q);

    Mat((Gt * P * Gt.t()) + Q).copyTo(P);///COVARIANCE PREDICTED

    S.copyTo(S_bar);

    //cout<<"Vt_R=" <<endl<<" "<<Vt_R<<endl<<endl;
    //cout<<"Gt="   <<endl<<" "<<Gt  <<endl<<endl;
    //cout<<"M="    <<endl<<" "<<M   <<endl<<endl;
    //cout<<"Q="    <<endl<<" "<<Q   <<endl<<endl;
    //cout<<"S="    <<endl<<" "<<S   <<endl<<endl;
    //cout<<"P="    <<endl<<" "<<P   <<endl<<endl;
    //cout<<"S_BAR="<<endl<<" "<<S   <<endl<<endl;
}

RState KalmanSLDM::predict_rob_pos(RState  rob_f0, double dt){
    RState d_rob;
    double v = rob_f0.vlin + 0.5 * rob_f0.alin * dt;
    double w = rob_f0.vang + 0.5 * rob_f0.aang * dt;
    if(fabs(w) > 0.001) {
        d_rob.xx   = v / w * ( - sin(rob_f0.xphi) + sin(rob_f0.xphi + w * dt));
        d_rob.xy   = v / w * (   cos(rob_f0.xphi) - cos(rob_f0.xphi + w * dt));
        d_rob.xphi =  rob_f0.vang * dt + 0.5 * rob_f0.aang * sqr(dt);
    } else {
        d_rob.xx   = v  * dt  * cos(rob_f0.xphi);
        d_rob.xy   = v  * dt  * sin(rob_f0.xphi);
        d_rob.xphi = 0.0;
    }
    d_rob.vlin =  rob_f0.alin * dt;
    d_rob.vang =  rob_f0.aang * dt;
    return d_rob;
}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::predict_rob(RState  rob_f0, double dt, Mat& Gt_R, Mat& Q){
    ///ROBOT STATE PREDICTION----

    RState d_rob = predict_rob_pos(rob_f0, dt);

    S.row(0) += d_rob.xx; S.row(1) += d_rob.xy; S.row(2) += normalizeAngle(d_rob.xphi);  S.row(3) += d_rob.vlin; S.row(4) += d_rob.vang;
    S.copyTo(S_bar);
    ///----ROBOT STATE PREDICTION

//    Mat Vt_R = Mat::zeros(rob_param, input_param, CV_64F);
    Vt_R = Mat::zeros(rob_param, input_param, CV_64F);
    ///ROBOT JACOBIANS----
    double v = rob_f0.vlin + 0.5 * rob_f0.alin * dt;
    double w = rob_f0.vang + 0.5 * rob_f0.aang * dt;
    if(fabs(w) > 0.001){

        Gt_R.row(0).col(2) =     - v   /     w  * (cos(rob_f0.xphi) - cos(rob_f0.xphi + w * dt));
        Gt_R.row(0).col(3) =     - 1.0 /     w  * (sin(rob_f0.xphi) - sin(rob_f0.xphi + w * dt));
        Gt_R.row(0).col(4) =       v   / sqr(w) * (cos(rob_f0.xphi + w * dt) * w * dt - sin(rob_f0.xphi + w * dt) + sin(rob_f0.xphi)) ;
        Gt_R.row(0).col(5) = - 0.5 * dt     / w * (sin(rob_f0.xphi) - sin(rob_f0.xphi + w * dt));
        Gt_R.row(0).col(6) =   0.5 * dt * v / w * (cos(rob_f0.xphi + w * dt) * w * dt - sin(rob_f0.xphi + w * dt) + sin(rob_f0.xphi)) ;

        Gt_R.row(1).col(2) =     - v   /     w  * (sin(rob_f0.xphi) - sin(rob_f0.xphi + w * dt));
        Gt_R.row(1).col(3) =     + 1.0 /     w  * (cos(rob_f0.xphi) - cos(rob_f0.xphi + w * dt));
        Gt_R.row(1).col(4) =     - v   / sqr(w) * (- sin(rob_f0.xphi + w * dt) * w * dt - cos(rob_f0.xphi + w * dt) + cos(rob_f0.xphi)) ;
        Gt_R.row(1).col(5) =     + 0.5 * dt / w * (cos(rob_f0.xphi) - cos(rob_f0.xphi + w * dt));
        Gt_R.row(1).col(6) = - 0.5 * dt * v / w * (- sin(rob_f0.xphi + w * dt) * w * dt - cos(rob_f0.xphi + w * dt) + cos(rob_f0.xphi)) ;
    }
    else{
        Gt_R.row(0).col(3) =   dt * cos(rob_f0.xphi);
        Gt_R.row(0).col(4) = - 0.5 * sqr(dt) * v * sin(rob_f0.xphi);//lim w->0 (v/w^2*(cos(a+w*t)*w*t-sin(a+w*t)+sin(a)))
        Gt_R.row(0).col(5) = - 0.5 * sqr(dt)     * cos(rob_f0.xphi);//lim w->0 (0.5 * t / w * (sin(a) - sin(a + w * t)))
        Gt_R.row(0).col(6) =   0.;                                  //lim w->0 (0.5*t*v/w*(cos(a+w*t)*w*t-sin(a+w*t)+sin(a)) )

        Gt_R.row(1).col(3) =   dt * sin(rob_f0.xphi);               //lim w->0 (+ 1.0 /     w  * (cos(a) - cos(a + w * t)))
        Gt_R.row(1).col(4) = - 0.5 * sqr(dt) * v * cos(rob_f0.xphi);//lim w->0 (v/w^2*(-sin(a+w*t)*w*t-cos(a+w*t)+cos(a)))
        Gt_R.row(1).col(5) =   0.5 * sqr(dt)     * sin(rob_f0.xphi);//lim w->0 (0.5 * t / w * (sin(a) - sin(a + w * t)))
        Gt_R.row(1).col(6) =   0.;                                  //lim w->0 (0.5*t*v/w*(-sin(a+w*t)*w*t-cos(a+w*t)+cos(a)))
    }
    Gt_R.row(2).col(4) = dt; Gt_R.row(2).col(6) = 0.5 * sqr(dt);
    Gt_R.row(3).col(5) = dt;
    Gt_R.row(4).col(6) = dt;

    Vt_R.row(0).col(0) = Gt_R.at<double>(0,3);
    Vt_R.row(1).col(0) = Gt_R.at<double>(1,3);
    Vt_R.row(0).col(1) = Gt_R.at<double>(0,4);
    Vt_R.row(1).col(1) = Gt_R.at<double>(1,4);

    Vt_R.row(2).col(1) = dt;
    Vt_R.row(3).col(0) = 1.;
    Vt_R.row(4).col(1) = 1.;
    ///----ROBOT JACOBIANS

    ///PREDICTION NOISE ----

    Vt_a = Mat::eye(rob_param,input_param,CV_64F);
    Vt_a.row(0).col(0) = Gt_R.at<double>(0,5);
    Vt_a.row(1).col(0) = Gt_R.at<double>(1,5);
    Vt_a.row(0).col(1) = Gt_R.at<double>(0,6);
    Vt_a.row(1).col(1) = Gt_R.at<double>(1,6);

    Vt_a.row(2).col(1) = 0.5 * sqr(dt);
    Vt_a.row(3).col(0) = dt;
    Vt_a.row(4).col(1) = dt;
    Vt_a.row(5).col(0) = 1.;
    Vt_a.row(6).col(1) = 1.;
    Mat M = Mat::zeros(input_param , input_param, CV_64F);
    if(dynamic_obj){
        M.row(0).col(0) = /*sqr(*/alfa_pre_rob_v_base /** dt * rob_f0.alin)*//* + 0.0001*/;// MULTIPLY WITH DT TODO
        M.row(1).col(1) = /*sqr(*/alfa_pre_rob_w_base /** dt * rob_f0.aang)*//* + 0.0001*/;

        Mat(Vt_a * M * Vt_a.t()).copyTo(Q.rowRange(0,rob_param).colRange(0,rob_param));
    }
    else{
        Mat(0.01 * Mat::eye(rob_param,rob_param,CV_64F)).copyTo(Q.rowRange(0,rob_param).colRange(0,rob_param));
        M.row(0).col(0) = 1 * /*sqr(*/alfa_pre_rob_v_base /** dt * rob_f0.alin)*//* + 0.0001*/;
        M.row(1).col(1) = 1 * /*sqr(*/alfa_pre_rob_w_base /** dt * rob_f0.aang)*//* + 0.0001*/;
        Mat(Vt_R * M * Vt_R.t()).copyTo(Q.rowRange(0,rob_param).colRange(0,rob_param));
    }


    //cout<<"qpredrob=" <<endl<<" "<<Vt_R * M * Vt_R.t()<<endl<<endl;
    //cout<<"vtr=" <<endl<<" "<<Vt_R<<endl<<endl;
    ///----PREDICTION NOISE
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::predict_obj(KInp u, Mat& Gt, Mat& Q){
    ///OBJECTS JACOBIANS & STATE PREDICTION----
    for(map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin();oi != Oi.end(); oi++){
        oi->first->life_time += u.dt;
        int i_min = oi->second.i_min;
        OiState obj_f0(oi->second.S_O);
        S.row(i_min + 3) += obj_f0.ax   * u.dt;
        S.row(i_min + 4) += obj_f0.ay   * u.dt;
        S.row(i_min + 5) += obj_f0.aphi * u.dt;
	

        if(dynamic_obj){
            Gt_Oi(u.dt) .copyTo(Gt.rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param));
            Mat(Q_Oi(alfa_pre_obj_xy_min, alfa_pre_obj_phi, u.dt)).copyTo(Q .rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param));
        }
        //cout<<"Vt_Oi_OO="<<endl<<" "<<Vt_Oi_OO<<endl<<endl;
        //cout<<"S_obj_predicted"<<endl<<" "<<Mat(S.rowRange(i_min,i_min+3))<<endl<<endl;
    }
    ///----OBJECTS JACOBIANS & STATE PREDICTION
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void KalmanSLDM::predict_p_cloud(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &input, RState  rob_f0, double dt){
    if(!input){ return; }

    for(typename std::vector<boost::shared_ptr<SegData> >::iterator inp = input->begin(); inp != input->end(); inp++){
        if(Oi.count((*inp)->getObj()) == 0){ continue; }

        OiState obj_f0(Oi[(*inp)->getObj()].S_O);
        OiState obj_f1;
	
	//TODO: still not quite ok; have to take into account that the relative pose changes after agent correction
	
	////////////////////////////////////////disabling point cloud prediction along symmetry direction
	cv::Mat Qm (Oi[(*inp)->getObj()].last_upd_info.Q); Qm.rowRange(0,2).colRange(0,2).copyTo(Qm);
	cv::Matx22d Mxr2w(cos(rob_f0.xphi), - sin(rob_f0.xphi), sin(rob_f0.xphi), cos(rob_f0.xphi));
	cv::Mat Mr2w(Mxr2w);
	Mat(Mr2w * Qm * Mr2w.t()).copyTo(Qm);
	
	
	bool symetry = true;
	
	cv::Mat_<double> eigval, eigvec, eigvec_cpy, eval; bool index_x; bool index_small;
	
	cv::eigen(Qm, eigval, eigvec);

	if(Qm.at<double>(0,0)>Qm.at<double>(1,1)){ index_x = 0; }
	else                                     { index_x = 1; }
	if     (fabs(eigval(0,!index_x))/*y*/ / fabs(eigval(0,index_x))/*x*/ > 20)      { index_small =  index_x; }
	else if(fabs(eigval(0,!index_x))/*y*/ / fabs(eigval(0,index_x))/*x*/ < 1.0/20.0){ index_small = !index_x; }
	else{ symetry = false; }
	
	if(symetry){
	    eval = cv::Mat(2,2,CV_64F,0.);
	    eval.row(0).col(0) = eigval.at<double>(0, index_x);
	    eval.row(1).col(1) = eigval.at<double>(0,!index_x);
	    eigvec.copyTo(eigvec_cpy);
	    if(index_x == 1){
		eigvec_cpy.colRange(0,2).row(1).copyTo(eigvec.colRange(0,2).row(0));
		eigvec_cpy.colRange(0,2).row(0).copyTo(eigvec.colRange(0,2).row(1));
	    }
	    Mat(eigvec.colRange(0,2).row(index_small)).copyTo(eigvec_cpy);
	    
	    obj_f0.vx = obj_f0.vx * eigvec_cpy.at<double>(0,0);
	    obj_f0.vy = obj_f0.vy * eigvec_cpy.at<double>(0,1);
	    obj_f0.ax = obj_f0.ax * eigvec_cpy.at<double>(0,0);
	    obj_f0.ay = obj_f0.ay * eigvec_cpy.at<double>(0,1);
	    
	    Mat(eigvec.colRange(0,2).row(!index_small)).copyTo(eigvec_cpy);
	    
	    Oi[(*inp)->getObj()].symmetry_flag = true;
	    Oi[(*inp)->getObj()].symmetry_vec_x = eigvec_cpy.at<double>(0,0);
	    Oi[(*inp)->getObj()].symmetry_vec_y = eigvec_cpy.at<double>(0,1);
	}
	else{
	    Oi[(*inp)->getObj()].symmetry_flag = false;
	}
	////////////////////////////////////////disabling point cloud prediction along symmetry direction

        obj_f1.vx   =   obj_f0.vx * cos(rob_f0.xphi) + obj_f0.vy * sin(rob_f0.xphi);
        obj_f1.vy   =   obj_f0.vy * cos(rob_f0.xphi) - obj_f0.vx * sin(rob_f0.xphi);
        obj_f1.ax   =   obj_f0.ax * cos(rob_f0.xphi) + obj_f0.ay * sin(rob_f0.xphi);
        obj_f1.ay   =   obj_f0.ay * cos(rob_f0.xphi) - obj_f0.ax * sin(rob_f0.xphi);

        xy t( obj_f1.vx * dt + obj_f1.ax * sqr(dt) / 2.0, obj_f1.vy * dt + obj_f1.ay * sqr(dt) / 2.0);
        for(PointDataVectorIter p = (*inp)->p.begin(); p != (*inp)->p.end(); p++){
            *p = PointData(to_polar( to_xy(*p) + t ));
        }
    }
}

template void KalmanSLDM::predict_p_cloud(SegmentDataPtrVectorPtr    &input, RState  rob_f0, double dt);
template void KalmanSLDM::predict_p_cloud(SegmentDataExtPtrVectorPtr &input, RState  rob_f0, double dt);

///------------------------------------------------------------------------------------------------------------------------------------------------///

cv::Mat KalmanSLDM::Gt_Oi(double dt){
    Mat Gt(Mat::eye(obj_param,obj_param,CV_64F));
    Gt.row(0).col(3) = dt; Gt.row(0).col(6) = sqr(dt) / 2.0;
    Gt.row(1).col(4) = dt; Gt.row(1).col(7) = sqr(dt) / 2.0;
    Gt.row(2).col(5) = dt; Gt.row(2).col(8) = sqr(dt) / 2.0;
    Gt.row(3).col(6) = dt;
    Gt.row(4).col(7) = dt;
    Gt.row(5).col(8) = dt;
    return Gt;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

cv::Mat KalmanSLDM::Q_Oi(double _obj_alfa_xy, double _obj_alpha_phi, double dt){
    _obj_alfa_xy   *= _obj_alfa_xy;
    _obj_alpha_phi *= _obj_alpha_phi;
    Mat Q(obj_param,obj_param,CV_64F,0.);
    //                    v--major diagonal--v                                                      v--top-diag area--v
    /*xx-xx*/    Q.row(0).col(0) = _obj_alfa_xy   * sqr(sqr(dt)) * dt / 20.0;/*xx-vx*/    Q.row(0).col(3) = _obj_alfa_xy   * sqr(sqr(dt)) / 8.0; /*xx-ax*/Q.row(0).col(6) = _obj_alfa_xy * sqr(dt) * dt / 6.0;
    /*xy-xy*/    Q.row(1).col(1) = _obj_alfa_xy   * sqr(sqr(dt)) * dt / 20.0;/*xy-vy*/    Q.row(1).col(4) = _obj_alfa_xy   * sqr(sqr(dt)) / 8.0; /*xy-ay*/Q.row(1).col(7) = _obj_alfa_xy * sqr(dt) * dt / 6.0;
    /*xphi-xphi*/Q.row(2).col(2) = _obj_alpha_phi * sqr(dt)      * dt /  3.0;/*xphi-vphi*/Q.row(2).col(5) = _obj_alpha_phi * sqr(dt)      / 2.0;
    /*vx-vx*/    Q.row(3).col(3) = _obj_alfa_xy   * sqr(dt)      * dt /  3.0;/*vx-ax*/    Q.row(3).col(6) = _obj_alfa_xy   * sqr(dt)      / 2.0;
    /*vy-vy*/    Q.row(4).col(4) = _obj_alfa_xy   * sqr(dt)      * dt /  3.0;/*vy-ay*/    Q.row(4).col(7) = _obj_alfa_xy   * sqr(dt)      / 2.0;
    /*vphi-vphi*/Q.row(5).col(5) = _obj_alpha_phi * dt;
    /*ax-ax*/    Q.row(6).col(6) = _obj_alfa_xy   * dt;
    /*ay-ay*/    Q.row(7).col(7) = _obj_alfa_xy   * dt;
    //    v--bottom-diag area--v
    /*vx-xx*/    Q.row(3).col(0) = _obj_alfa_xy   * sqr(sqr(dt)) / 8.0;
    /*vy-xy*/    Q.row(4).col(1) = _obj_alfa_xy   * sqr(sqr(dt)) / 8.0;
    /*vphi-xphi*/Q.row(5).col(2) = _obj_alpha_phi * sqr(dt)      / 2.0;
    /*ax-xx*/    Q.row(6).col(0) = _obj_alfa_xy   * sqr(dt) * dt / 6.0;
    /*ay-xy*/    Q.row(7).col(1) = _obj_alfa_xy   * sqr(dt) * dt / 6.0;
    /*ax-vx*/    Q.row(6).col(3) = _obj_alfa_xy   * sqr(dt)      / 2.0;
    /*ay-vy*/    Q.row(7).col(4) = _obj_alfa_xy   * sqr(dt)      / 2.0;
    return Q;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
