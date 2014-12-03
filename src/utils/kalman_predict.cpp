#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;


///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::prediction(SegmentDataPtrVectorPtr &input, KInp &u){
    //cout<<"prediction:"<<endl;
    if(u.v != u.v){ u.v = 0; }//correcting NaN stuff
    if(u.w != u.w){ u.w = 0; }

    Mat Gt (Mat::zeros(P.rows, P.cols, CV_64F));
    Mat Gt_R = Gt.rowRange(0, rob_param).colRange(0, rob_param); Mat(Mat::eye(3,3,CV_64F)).copyTo(Gt_R);

    Mat Q(P.rows,P.cols,CV_64F, 0.);
    RState  rob_f0(S);

    predict_p_cloud(input, rob_f0, u);
    predict_obj(u,Gt, Q);

    predict_rob(rob_f0, u, Gt_R, Q);

    Mat((Gt * P * Gt.t()) + Q).copyTo(P);///COVARIANCE PREDICTED

    //cout<<"Vt_R="<<endl<<" "<<Vt_R<<endl<<endl;
    //cout<<"Gt="  <<endl<<" "<<Gt  <<endl<<endl;
    //cout<<"M="   <<endl<<" "<<M   <<endl<<endl;
    //cout<<"Q="   <<endl<<" "<<Q   <<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::predict_rob(RState  rob_f0, KInp u, Mat& Gt_R, Mat& Q){
    ///ROBOT STATE PREDICTION----
    double dx, dy, da;

    if(fabs(u.w) > 0.001) {
        dx = (- u.v/u.w * sin(rob_f0.xphi) + u.v/u.w * sin(rob_f0.xphi + u.w * u.dt));
        dy = (+ u.v/u.w * cos(rob_f0.xphi) - u.v/u.w * cos(rob_f0.xphi + u.w * u.dt));
        da = u.w * u.dt;
    } else {
        dx = u.v  * u.dt  * cos(rob_f0.xphi);
        dy = u.v  * u.dt  * sin(rob_f0.xphi);
        da = 0.0;
    }
    S.row(0) += dx; S.row(1) += dy; S.row(2) += da;  Mat(S.rowRange(0,3)).copyTo(S_R_bar);
    ///----ROBOT STATE PREDICTION


    Mat Vt_R = Mat::eye(rob_param, input_param, CV_64F);
    ///ROBOT JACOBIANS----
    if(fabs(u.w) > 0.001){
        Gt_R.row(0).col(2) = - (u.v * (cos(rob_f0.xphi) - cos(u.dt * u.w + rob_f0.xphi)) / u.w);
        Gt_R.row(1).col(2) = - (u.v * (sin(rob_f0.xphi) - sin(u.dt * u.w + rob_f0.xphi)) / u.w);

        Vt_R.row(0).col(0) = - ((sin(rob_f0.xphi) - sin(u.dt * u.w + rob_f0.xphi)) / u.w);
        Vt_R.row(1).col(0) =    (cos(rob_f0.xphi) - cos(u.dt * u.w + rob_f0.xphi)) / u.w;
        Vt_R.row(0).col(1) = u.v * (cos(u.dt * u.w + rob_f0.xphi) * u.dt * u.w + sin(rob_f0.xphi) - sin(u.dt * u.w + rob_f0.xphi)) / sqr(u.w);
        Vt_R.row(1).col(1) = u.v * (sin(u.dt * u.w + rob_f0.xphi) * u.dt * u.w - cos(rob_f0.xphi) + cos(u.dt * u.w + rob_f0.xphi)) / sqr(u.w);
        Vt_R.row(2).col(1) = u.dt;
    }
    else{
        Vt_R.row(0).col(1) = -0.5 * sqr(u.dt) * u.v * sin(rob_f0.xphi);
        Vt_R.row(1).col(1) =  0.5 * sqr(u.dt) * u.v * cos(rob_f0.xphi);
        Vt_R.row(0).col(0) = u.dt * cos(rob_f0.xphi);
        Vt_R.row(1).col(0) = u.dt * sin(rob_f0.xphi);
        Vt_R.row(2).col(1) = u.dt;
    }
//        Vt_R.row(0).col(2) = u.v * cos(u.dt * u.w + rob_f0.xphi);
//        Vt_R.row(1).col(2) = u.v * sin(u.dt * u.w + rob_f0.xphi);
//        Vt_R.row(2).col(2) = u.w;
    ///----ROBOT JACOBIANS


    //////INPUT NOISE ----
    double sign;
    if(fabs(u.v) > fabs(v_static)){ sign = sgn(u.v); }
    else{                           sign = sgn(v_static);}
    double v_noise = sign * fmax(fabs(u.v), fabs(v_static));
    v_static = (v_static + u.v) / 2.0;
    if(fabs(u.w) > fabs(w_static)){ sign = sgn(u.w); }
    else{                           sign = sgn(w_static);}
    double w_noise = sign * fmax(fabs(u.w), fabs(w_static));
    w_static = (w_static + u.w) / 2.0;


    Mat M = Mat::zeros(input_param , input_param, CV_64F);
    M.row(0).col(0) = input_noise[0]*sqr(v_noise) + input_noise[1] * sqr(w_noise);
    M.row(1).col(1) = input_noise[2]*sqr(v_noise) + input_noise[3] * sqr(w_noise);
    M.row(2).col(2) = input_noise[4]*sqr(u.dt);

    Mat(Vt_R * M * Vt_R.t()).copyTo(Q.rowRange(0,rob_param).colRange(0,rob_param));
    ///----INPUT NOISE
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::predict_obj(KInp u, Mat& Gt, Mat& Q){
    ///OBJECTS JACOBIANS & STATE PREDICTION----
    for(map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin();oi != Oi.end(); oi++){
        int i_min = oi->second.i_min;
        OiState obj_f0(oi->second.S_O);
        //velocity predicted in segmentation
        S.row(i_min + 3) += obj_f0.ax   * u.dt;
        S.row(i_min + 4) += obj_f0.ay   * u.dt;
        S.row(i_min + 5) += obj_f0.aphi * u.dt;

        Gt_Oi(u.dt).copyTo(Gt.rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param));
        Mat(Q_Oi(u.dt)).copyTo(Q.rowRange(i_min, i_min + obj_param).colRange(i_min, i_min + obj_param));

        //cout<<"Vt_Oi_OO="<<endl<<" "<<Vt_Oi_OO<<endl<<endl;
        //cout<<"S_obj_predicted"<<endl<<" "<<Mat(S.rowRange(i_min,i_min+3))<<endl<<endl;
    }
    ///----OBJECTS JACOBIANS & STATE PREDICTION

}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::predict_p_cloud(SegmentDataPtrVectorPtr &input, RState  rob_f0, KInp u){
    if(!input){
        return;
    }
    for(SegmentDataPtrVector::iterator inp = input->begin(); inp != input->end(); inp++){
        if(Oi.count((*inp)->getObj()) == 0){
            continue;
        }
        double dt = u.dt;//ALL THIS TF HAS TO BE TF-ED ITSELF IN NEW ROB_BAR FRAME
        OiState obj_f0(Oi[(*inp)->getObj()].S_O);
        OiState obj_f1;


        obj_f1.xx   =  (obj_f0.xx - rob_f0.xx) * cos(rob_f0.xphi) + (obj_f0.xy - rob_f0.xy) * sin(rob_f0.xphi);
        obj_f1.xy   =  (obj_f0.xy - rob_f0.xy) * cos(rob_f0.xphi) - (obj_f0.xx - rob_f0.xx) * sin(rob_f0.xphi);
        obj_f1.xphi =   obj_f0.xphi - rob_f0.xphi;
        obj_f1.vx   =   obj_f0.vx * cos(rob_f0.xphi) + obj_f0.vy * sin(rob_f0.xphi);
        obj_f1.vy   =   obj_f0.vy * cos(rob_f0.xphi) - obj_f0.vx * sin(rob_f0.xphi);
        obj_f1.vphi =   obj_f0.vphi;
        obj_f1.ax   =   obj_f0.ax * cos(rob_f0.xphi) + obj_f0.ay * sin(rob_f0.xphi);
        obj_f1.ay   =   obj_f0.ay * cos(rob_f0.xphi) - obj_f0.ax * sin(rob_f0.xphi);
        obj_f1.aphi =   obj_f0.aphi;

        xy com_x_f1(obj_f1.xx/*(*inp)->parrent->com.x*/, obj_f1.xy/*(*inp)->parrent->com.y*/);

        xy t;double angle;
        t.x = obj_f1.vx * dt + obj_f1.ax * sqr(dt) / 2.0;
        t.y = obj_f1.vy * dt + obj_f1.ay * sqr(dt) / 2.0;
        angle = 0;// /*obj_xphi_f1 + */(obj_f1.vphi * dt + obj_f1.aphi * sqr(dt) / 2.0);

        cv::Mat_<double>T = (cv::Mat_<double>(3,3) << cos(angle), -sin(angle), t.x,
                                                       sin(angle),  cos(angle),t.y,
                                                                0,           0, 1);
        cv::Matx33d Tt(T);
        for(PointDataVectorIter p = (*inp)->p.begin(); p != (*inp)->p.end(); p++){
            *p = PointData(to_polar(mat_mult(Tt, to_xy(*p) - com_x_f1)+com_x_f1));
        }
    }
}

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

cv::Mat KalmanSLDM::Q_Oi(double dt){
    Mat Q(obj_param,obj_param,CV_64F,0.);
    //                    v--major diagonal--v                                                      v--top-diag area--v
    /*xx-xx*/    Q.row(0).col(0) = obj_noise_x   * sqr(sqr(dt)) * dt / 20.0;/*xx-vx*/    Q.row(0).col(3) = obj_noise_x   * sqr(sqr(dt)) / 8.0; /*xx-ax*/Q.row(0).col(6) = obj_noise_x * sqr(dt) * dt / 6.0;
    /*xy-xy*/    Q.row(1).col(1) = obj_noise_y   * sqr(sqr(dt)) * dt / 20.0;/*xy-vy*/    Q.row(1).col(4) = obj_noise_y   * sqr(sqr(dt)) / 8.0; /*xy-ay*/Q.row(1).col(7) = obj_noise_y * sqr(dt) * dt / 6.0;
    /*xphi-xphi*/Q.row(2).col(2) = obj_noise_phi * sqr(dt)      * dt /  3.0;/*xphi-vphi*/Q.row(2).col(5) = obj_noise_phi * sqr(dt)      / 2.0;
    /*vx-vx*/    Q.row(3).col(3) = obj_noise_x   * sqr(dt)      * dt /  3.0;/*vx-ax*/    Q.row(3).col(6) = obj_noise_x   * sqr(dt)      / 2.0;
    /*vy-vy*/    Q.row(4).col(4) = obj_noise_y   * sqr(dt)      * dt /  3.0;/*vy-ay*/    Q.row(4).col(7) = obj_noise_y   * sqr(dt)      / 2.0;
    /*vphi-vphi*/Q.row(5).col(5) = obj_noise_phi * dt;
    /*ax-ax*/    Q.row(6).col(6) = obj_noise_x   * dt;
    /*ay-ay*/    Q.row(7).col(7) = obj_noise_y   * dt;
    //    v--bottom-diag area--v
    /*vx-xx*/    Q.row(3).col(0) = obj_noise_x   * sqr(sqr(dt)) / 8.0;
    /*vy-xy*/    Q.row(4).col(1) = obj_noise_y   * sqr(sqr(dt)) / 8.0;
    /*vphi-xphi*/Q.row(5).col(2) = obj_noise_phi * sqr(dt)      / 2.0;
    /*ax-xx*/    Q.row(6).col(0) = obj_noise_x   * sqr(dt) * dt / 6.0;
    /*ay-xy*/    Q.row(7).col(1) = obj_noise_y   * sqr(dt) * dt / 6.0;
    /*ax-vx*/    Q.row(6).col(3) = obj_noise_x   * sqr(dt)      / 2.0;
    /*ay-vy*/    Q.row(7).col(4) = obj_noise_y   * sqr(dt)      / 2.0;
    return Q;
}
