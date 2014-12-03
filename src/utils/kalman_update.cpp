#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;


///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init_Oi(ObjectDataPtr obj, xy obj_com_bar_f1){
    RState rob_bar_f0(S);
    if(Oi.count(obj) == 0){ return; }
    int i_min = Oi[obj].i_min;

    S.row(i_min + 0) = rob_bar_f0.xx + obj_com_bar_f1.x * cos(rob_bar_f0.xphi) - obj_com_bar_f1.y * sin(rob_bar_f0.xphi);
    S.row(i_min + 1) = rob_bar_f0.xy + obj_com_bar_f1.y * cos(rob_bar_f0.xphi) + obj_com_bar_f1.x * sin(rob_bar_f0.xphi);
    S.row(i_min + 2) = rob_bar_f0.xphi ;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::update_Oi(ObjectDataPtr seg, KObjZ kObjZ){
    if(add_obj(seg, kObjZ)){ return; }
    //cout<<"innovation:"<<endl;

    RState  rob_bar_f0(S);
    OiState obj_bar_f0(Oi[seg].S_O);

    ///PREDICTED OBSERVATION----
    Mat h_bar_f1(z_param, 1, CV_64F, 0.0);
    h_bar_f1.row(0) = (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) + (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi);
    h_bar_f1.row(1) = (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) - (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi);
    h_bar_f1.row(2) =  obj_bar_f0.xphi - rob_bar_f0.xphi;
    ///----PREDICTED OBSERVATION

    ///PREDICTED OBSERVATION JACOBIAN AND NOISE----
    Mat Ht_low(Mat(z_param, rob_param + obj_param, CV_64F, 0.));

    Ht_low.row(0).col(0) = - cos(rob_bar_f0.xphi); Ht_low.row(0).col(1) = - sin(rob_bar_f0.xphi);
    Ht_low.row(1).col(0) =   sin(rob_bar_f0.xphi); Ht_low.row(1).col(1) = - cos(rob_bar_f0.xphi) ;
    Ht_low.row(2).col(2) = - 1.;

    Ht_low.row(0).col(2) = - (obj_bar_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi) + (obj_bar_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi);
    Ht_low.row(1).col(2) = - (obj_bar_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi) - (obj_bar_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi);

    Ht_low.row(0).col(3) =   cos(rob_bar_f0.xphi); Ht_low.row(0).col(4) = sin(rob_bar_f0.xphi);
    Ht_low.row(1).col(3) = - sin(rob_bar_f0.xphi); Ht_low.row(1).col(4) = cos(rob_bar_f0.xphi) ;
    Ht_low.row(2).col(5) =  1.;

    Mat Ht = Ht_low * Fxi(seg);

    Mat Q(kObjZ.Q);
    ///----PREDICTED OBSERVATION JACOBIAN AND NOISE


    Mat h_hat_f1(z_param, 1, CV_64F, 0.0);
    h_hat_f1.row(0) = kObjZ.pos.x; h_hat_f1.row(1) = kObjZ.pos.y; h_hat_f1.row(2) = kObjZ.phi; ///OBSERVATION


    Mat Kt = P * Ht.t() * ( Ht * P * Ht.t() + Q ).inv();                ///KALMAN GAIN
    Mat(S + Kt * ( h_hat_f1 - h_bar_f1 )).copyTo(S);                    ///STATE UPDATE
    Mat((cv::Mat::eye(P.rows, P.cols, CV_64F) - Kt * Ht) * P).copyTo(P);///COVARIANCE UPDATE

    //cout<<"h_hat_f1=" <<endl<<" "<<h_hat_f1                        <<endl<<endl;
    //cout<<"h_bar_f1=" <<endl<<" "<<h_bar_f1                        <<endl<<endl;
    //cout<<"h_diff_f1="<<endl<<" "<<Mat(h_hat_f1 - h_bar_f1)        <<endl<<endl;
    //cout<<"Kt="       <<endl<<" "<<Kt                              <<endl<<endl;
    //cout<<"Kt*dh"     <<endl<<" "<<Mat(Kt * (h_hat_f1 - h_bar_f1 ))<<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
}