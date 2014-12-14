#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;

///------------------------------------------------------------------------------------------------------------------------------------------------///

KalmanSLDM::KalmanSLDM(RecfgParam& _param, SensorTf& _tf_sns) :
    tf_sns(_tf_sns),
    rob_alfa_1(_param.kalman_rob_alfa_1),
    rob_alfa_2(_param.kalman_rob_alfa_2),
    rob_alfa_3(_param.kalman_rob_alfa_3),
    rob_alfa_4(_param.kalman_rob_alfa_4),
    obj_alfa_xy_min(_param.kalman_obj_alfa_xy_min),
    obj_alfa_xy_max(_param.kalman_obj_alfa_xy_max),
    obj_alfa_max_vel(_param.kalman_obj_alfa_max_vel),
    obj_alfa_phi(_param.kalman_obj_alfa_phi),
    obj_init_pow_dt(_param.kalman_obj_init_pow_dt),
    obj_timeout(_param.kalman_obj_timeout),
    discard_old_seg_perc(_param.kalman_discard_old_seg_perc){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init(RState rob_x){
    pos_init = false;
    S.release(); S_bar.release(); P.release();;
    Oi.clear();
    v_static = 0;
    w_static = 0;

    S_bar.push_back(0.);       S_bar.push_back(0.);       S_bar.push_back(0.);
    S    .push_back(rob_x.xx); S    .push_back(rob_x.xy); S    .push_back(rob_x.xphi);
    P = Mat::zeros(rob_param, rob_param, CV_64F);

    S_bar_old.push_back(0.);       S_bar_old.push_back(0.);       S_bar_old.push_back(0.);
    S_old    .push_back(rob_x.xx); S_old    .push_back(rob_x.xy); S_old    .push_back(rob_x.xphi);
    P_old = Mat::zeros(rob_param, rob_param, CV_64F);

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

    S.push_back(cos(rob_bar_f0.xphi) * obj_zhat_f1.xx -
                sin(rob_bar_f0.xphi) * obj_zhat_f1.xy +
                cos(tf_sns.getPhi()) * (rob_bar_f0.xx - tf_sns.getXY().x + tf_sns.getXY().x * cos(rob_bar_f0.xphi) - tf_sns.getXY().y * sin(rob_bar_f0.xphi)) +
                sin(tf_sns.getPhi()) * (rob_bar_f0.xy - tf_sns.getXY().y + tf_sns.getXY().y * cos(rob_bar_f0.xphi) + tf_sns.getXY().x * sin(rob_bar_f0.xphi)));
    S.push_back(cos(rob_bar_f0.xphi) * obj_zhat_f1.xy +
                sin(rob_bar_f0.xphi) * obj_zhat_f1.xx +
                cos(tf_sns.getPhi()) * (   rob_bar_f0.xy - tf_sns.getXY().y + tf_sns.getXY().y * cos(rob_bar_f0.xphi) + tf_sns.getXY().x * sin(rob_bar_f0.xphi)) +
                sin(tf_sns.getPhi()) * ( - rob_bar_f0.xx + tf_sns.getXY().x - tf_sns.getXY().x * cos(rob_bar_f0.xphi) + tf_sns.getXY().y * sin(rob_bar_f0.xphi)));

//    S.push_back(rob_bar_f0.xx   + obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi));
//    S.push_back(rob_bar_f0.xy   + obj_zhat_f1.xy * cos(rob_bar_f0.xphi) + obj_zhat_f1.xx * sin(rob_bar_f0.xphi));
    S.push_back(rob_bar_f0.xphi + obj_zhat_f1.xphi);
    S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 ); S.push_back( 0.0 );

    hconcat(P, Mat(P.rows, obj_param, CV_64F, 0.), P);
    P.resize(P.cols,0.);//resize-ing P
    update_sub_mat();   //re-binding sub-matrices masks

    Oi[seg].i_min = P.rows - obj_param;//mapping object specific information
    Oi[seg].S_O   = S.rowRange(S.rows - obj_param, S.rows);
    Mat P_ORi(P_OR.rowRange(P_OR.rows - obj_param, P_OR.rows));
    Mat P_ROi(P_RO.colRange(P_RO.cols - obj_param, P_RO.cols));
    Oi[seg].P_OO  = P.rowRange(P.rows - obj_param, P.rows).colRange(P.cols - obj_param, P.cols);


    ///OBJECT COVARIANCE INITIALIZATION----
    Mat Gt_hinv_R (obj_param, rob_param, CV_64F, 0.);

    Gt_hinv_R.row(0).col(0) =   cos(tf_sns.getPhi()); Gt_hinv_R.row(0).col(1) = sin(tf_sns.getPhi());
    Gt_hinv_R.row(1).col(0) = - sin(tf_sns.getPhi()); Gt_hinv_R.row(1).col(1) = cos(tf_sns.getPhi());
    Gt_hinv_R.row(2).col(2) = 1.;

    Gt_hinv_R.row(0).col(2) = - obj_zhat_f1.xy * cos(rob_bar_f0.xphi) - obj_zhat_f1.xx * sin(rob_bar_f0.xphi) + //only this is wihtout sensor
                                cos(tf_sns.getPhi()) * ( - tf_sns.getXY().y * cos(rob_bar_f0.xphi) - tf_sns.getXY().x * sin(rob_bar_f0.xphi)) +
                                sin(tf_sns.getPhi()) * (   tf_sns.getXY().x * cos(rob_bar_f0.xphi) - tf_sns.getXY().y * sin(rob_bar_f0.xphi));
    Gt_hinv_R.row(1).col(2) =   obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi) + //only this is wihtout sensor
                                cos(tf_sns.getPhi()) * (   tf_sns.getXY().x * cos(rob_bar_f0.xphi) - tf_sns.getXY().y * sin(rob_bar_f0.xphi)) +
                                sin(tf_sns.getPhi()) * (   tf_sns.getXY().y * cos(rob_bar_f0.xphi) + tf_sns.getXY().x * sin(rob_bar_f0.xphi));;

    Mat P_OO_V(9, 9, CV_64F, 0.);

//    Mat Gt_hinv_Z (obj_param, z_param, CV_64F, 0.);
//    Gt_hinv_Z.row(0).col(0) = cos(rob_bar_f0.xphi); Gt_hinv_Z.row(0).col(1) = - sin(rob_bar_f0.xphi);
//    Gt_hinv_Z.row(1).col(0) = sin(rob_bar_f0.xphi); Gt_hinv_Z.row(1).col(1) =   cos(rob_bar_f0.xphi);
//    Gt_hinv_Z.row(2).col(2) = 1.;

    Q_Oi(obj_alfa_xy_min, obj_alfa_phi, obj_init_pow_dt).copyTo(P_OO_V);

    Mat(Gt_hinv_R * P.rowRange(0, rob_param).colRange(0, rob_param) * Gt_hinv_R.t()  + P_OO_V/*Gt_hinv_Z * Mat(kObjZ.Q)  * Gt_hinv_Z.t()*/).copyTo(Oi[seg].P_OO);

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
    for(int i = 0;i < rob_param; i++){
        Fx.row(i).col(i) = 1.;
    }
    for(int i = 0;i < obj_param; i++){
        Fx.row(rob_param + i).col(Oi[seg].i_min + i) = 1.;
    }
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
