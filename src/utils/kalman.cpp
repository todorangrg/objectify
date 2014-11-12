#include "utils/kalman.h"
#include "data_processing/correlation.h"

using namespace cv;
using namespace std;
//TODO include displacement of the sensor in the model

OiState::OiState(cv::Mat _S_O):
    xx(_S_O.at<double>(0)), xy(_S_O.at<double>(1)), xphi(_S_O.at<double>(2)),
    vx(_S_O.at<double>(3)), vy(_S_O.at<double>(4)), vphi(_S_O.at<double>(5)),
    ax(_S_O.at<double>(6)), ay(_S_O.at<double>(7)), aphi(_S_O.at<double>(8)){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

RState::RState(cv::Mat _S):
    xx(_S.at<double>(0)), xy(_S.at<double>(1)), xphi(_S.at<double>(2)){}

///------------------------------------------------------------------------------------------------------------------------------------------------///


//TODO : THIS IS VERY INEFFICIENT AND NINJA DONE, MAKE IT "NEAT"


class NeighDatasimple{
public:
    SegmentDataPtr neigh;
    double         prob_fwd;
    double         prob_rev;
    NeighDatasimple(SegmentDataPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev){}
};



void KalmanSLDM::advance(SensorData& sensor, bool advance){//TODO deal with "this only if "timeout done / covariance of reprediction too big" : check it in the old sensor frame, in the no_innovation flag and move them in new frame
    if(advance){
        if(!pos_init){
            return;
        }
        S_R_bar_old = Mat(S_R_bar.rows,S_R_bar.cols,CV_64F); S_R_bar.copyTo(S_R_bar_old);
        S_old = Mat(S.rows,S.cols,CV_64F); S.copyTo(S_old);
        P_old = Mat(P.rows,P.cols,CV_64F); P.copyTo(P_old);
        Oi_old = Oi;
        for(int j=0;j<sensor.frame_new->seg_ext->size();j++){
            for(int i=0;i<adv_erase_obj.size();i++){
                if( adv_erase_obj.at(i) ==  sensor.frame_new->seg_ext->at(j)->parrent->parrent){
                    for(int k=0;k<sensor.objects->size();k++){
                        if( adv_erase_obj.at(i) ==  sensor.objects->at(j)){
                            sensor.objects->erase(sensor.objects->begin() + k);
                            break;
                        }
                    }
                    break;
                }
            }
            sensor.frame_new->seg_ext->at(j)->parrent->parrent.reset();
            if(seg_ext_new_obj.count(sensor.frame_new->seg_ext->at(j)) != 0){ //assigning new parrent value
                sensor.frame_new->seg_ext->at(j)->parrent->parrent = seg_ext_new_obj[sensor.frame_new->seg_ext->at(j)];
            }
            else{
                int ind_er = 0;
                for(int i=0;i<sensor.frame_new->seg_init->size();i++){
                    if(sensor.frame_new->seg_ext->at(j)->parrent == sensor.frame_new->seg_init->at(i)){
                        ind_er = i;
                    }
                }
                sensor.frame_new->seg_init->erase(sensor.frame_new->seg_init->begin() + ind_er);
            }
        }
        for(int i=0;i<adv_no_innv_seg.size();i++){
            std::vector<NeighDatasimple> neighbours;
            for(PointDataVectorIter   p_ref=adv_no_innv_seg.at(i)->p.begin();p_ref!=adv_no_innv_seg.at(i)->p.end();p_ref++){
                double circle_rad = 0.5;//////?!!!!!!!!!!!!!!!!!
                double ang_bounds[2];
                angular_bounds(*p_ref,circle_rad, ang_bounds);
                SegmentDataPtrVectorIter seg_min = sensor.frame_new->seg_init->end();
                bool found_one = false;
                for(int k=0;k<sensor.frame_new->seg_init->size();k++){
                    for(int kk = 0;kk<sensor.frame_new->seg_init->at(k)->p.size();kk++){
                        if( diff( sensor.frame_new->seg_init->at(k)->p.at(kk) , *p_ref ) <= circle_rad ){
                            circle_rad = diff( sensor.frame_new->seg_init->at(k)->p.at(kk) , *p_ref );//retine iteratorul minim intro var sa o inserezi jos vv
                            seg_min = sensor.frame_new->seg_init->begin() + k;
                            found_one=true;
                        }
                    }
                    if(found_one){
                        std::vector<NeighDatasimple>::iterator it_neigh_data = neighbours.begin();
                        while(it_neigh_data != neighbours.end() ){
                            if( *seg_min == it_neigh_data->neigh ){
                                it_neigh_data->prob_fwd += 1.0 / (double)adv_no_innv_seg.at(i)->p.size() ;
                                break;
                            }
                            it_neigh_data++;
                        }
                        if( it_neigh_data == neighbours.end() ){
                            neighbours.push_back(NeighDatasimple(*seg_min, 1.0 / (double)adv_no_innv_seg.at(i)->p.size(), 0.0));
                        }
                    }
                }
                for(int k=0;k<neighbours.size();k++){
                    if(neighbours.at(k).prob_fwd > 0.9){
                        for(int kk=0;kk<sensor.frame_new->seg_init->size();kk++){
                            if(neighbours.at(k).neigh == sensor.frame_new->seg_init->at(kk)){
                                sensor.frame_new->seg_init->erase(sensor.frame_new->seg_init->begin() + kk);
                                break;
                            }
                        }
                        break;
                    }
                }

            }
        }
        for(int i=0;i<adv_no_innv_seg.size();i++){
            sensor.frame_new->seg_init->push_back(adv_no_innv_seg.at(i));
        }
    }
    else{
        S_R_bar = Mat(S_R_bar_old.rows,S_R_bar_old.cols,CV_64F); S_R_bar_old.copyTo(S_R_bar);
        S = Mat(S_old.rows,S_old.cols,CV_64F); S_old.copyTo(S);
        P = Mat(P_old.rows,P_old.cols,CV_64F); P_old.copyTo(P);
        Oi = Oi_old;
        update_sub_mat();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::run(SensorData& sensor, std::map <SegmentDataExtPtr, std::vector<NeighData> >& neigh_data_to, std::map <SegmentDataExtPtr, std::vector<NeighData> >& neigh_data_tn){
    adv_erase_obj.clear();
    seg_ext_new_obj.clear();
    adv_no_innv_seg.clear();

    if(sensor.status != OLD_FRAME){ return; }
    KControl u(sensor.frame_new->rob_vel.v, sensor.frame_new->rob_vel.w, sensor.frame_old->past_time.toSec());
    if(u.dt < 0.001){ return; }
    if(!pos_init){ pos_init = true;
        S.row(0) = sensor.frame_old->rob_x.x; S.row(1) = sensor.frame_old->rob_x.y; S.row(2) = sensor.frame_old->rob_x.angle;
    }

    prediction(u);

    for(int i=0;i<sensor.frame_new->seg_ext->size();i++){// remember future erase of all objects of segments that have neigh
        if(sensor.frame_new->seg_ext->at(i)->corr_flag != CORR_NEWOBJ){
            adv_erase_obj.push_back(sensor.frame_new->seg_ext->at(i)->parrent->parrent);
        }
    }


    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg_tn = neigh_data_tn.begin();it_seg_tn != neigh_data_tn.end();it_seg_tn++){
        if(it_seg_tn->first->corr_flag == CORR_NEWOBJ){
            seg_ext_new_obj[it_seg_tn->first] = it_seg_tn->first->parrent->parrent;
            //.....
            continue;
        }

        for(std::vector<NeighData>::iterator it_seg_to = it_seg_tn->second.begin();it_seg_to != it_seg_tn->second.end();it_seg_to++){
            if(((it_seg_tn->first->corr_flag == CORR_121)&&(it_seg_to->neigh->corr_flag == CORR_121))||
               ((it_seg_tn->first->corr_flag == CORR_121)&&(it_seg_to->neigh->corr_flag == CORR_12MANY))){
                if((it_seg_tn->first->corr_flag == CORR_121)&&(it_seg_to->neigh->corr_flag == CORR_12MANY)){
                    std::cout<<" 1 t-1 to many"<<std::endl;
                }
                //simple 121 or // one t-1 to many t
                seg_ext_new_obj[it_seg_tn->first] = it_seg_to->neigh->parrent->parrent;

                ConvData conv = *it_seg_to->neigh->conv;
                TFdata tf = conv.tf->front();
                xy     com_old = conv.com;               //old com in rob_bar frame after motion prediction
                xy     com_new = com_old + tf.tf.xy_mean;//new com in rob_bar frame after motion prediction
                double phi_new = 0.0 + tf.tf.ang_mean;

                KObjZ kObjZ; kObjZ.pos = com_new; kObjZ.phi = phi_new; kObjZ.Q = cv::Matx33d::zeros();
                kObjZ.Q(0,0) = tf.tf.xy_cov(0,0); kObjZ.Q(0,1) = tf.tf.xy_cov(0,1); kObjZ.Q(1,0) = tf.tf.xy_cov(1,0); kObjZ.Q(1,1) = tf.tf.xy_cov(1,1); kObjZ.Q(2,2) = tf.tf.ang_cov;//all this is in rob_bar frame


                init_Oi(it_seg_to->neigh->parrent->parrent,com_old);
                update_Oi(it_seg_to->neigh->parrent->parrent, kObjZ);
            }
            else if((it_seg_tn->first->corr_flag == CORR_MANY21)&&(it_seg_to->neigh->corr_flag == CORR_121)){
                // many t-1 to one t
                ConvData conv = *it_seg_to->neigh->conv;
                TFdata tf = conv.tf->front();
                xy     com_old = conv.com;               //old com in rob_bar frame after motion prediction
                xy     com_new = com_old + tf.tf.xy_mean;//new com in rob_bar frame after motion prediction
                double phi_new = 0.0 + tf.tf.ang_mean;

                KObjZ kObjZ; kObjZ.pos = com_new; kObjZ.phi = phi_new; kObjZ.Q = cv::Matx33d::zeros();
                kObjZ.Q(0,0) = tf.tf.xy_cov(0,0); kObjZ.Q(0,1) = tf.tf.xy_cov(0,1); kObjZ.Q(1,0) = tf.tf.xy_cov(1,0); kObjZ.Q(1,1) = tf.tf.xy_cov(1,1); kObjZ.Q(2,2) = tf.tf.ang_cov;//all this is in rob_bar frame


                init_Oi(it_seg_to->neigh->parrent->parrent,com_old);
                update_Oi(it_seg_to->neigh->parrent->parrent, kObjZ);
                if(it_seg_to == it_seg_tn->second.begin()){
                    seg_ext_new_obj[it_seg_tn->first] = it_seg_to->neigh->parrent->parrent;
                }
                else if(it_seg_tn->second.begin()->neigh->parrent->parrent != it_seg_to->neigh->parrent->parrent){
                    //update it_seg_tn->second.begin() with it_seg_to
                    //delete it_seg_to
                    update_Oi_with_Oj(it_seg_tn->second.begin()->neigh->parrent->parrent,it_seg_to->neigh->parrent->parrent);
                    rmv_obj(it_seg_to->neigh->parrent->parrent);
                }

                //here find all from t-1; update first found with all the others, using their S as a "full" observation ( no rob update, no jacobian) and their P as observation noise
                //test how merging of lines with "fake" velocity works

                std::cout<<" many t-1 to 1"<<std::endl;
            }
            else if((it_seg_tn->first->corr_flag == CORR_MANY21)&&(it_seg_to->neigh->corr_flag == CORR_12MANY)){
                std::cout<<" ERROR : kalman recieved many 2 many object situation"<<std::endl;
            }
        }
    }



    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg_to = neigh_data_to.begin();it_seg_to != neigh_data_to.end();it_seg_to++){
        if(it_seg_to->first->corr_flag == CORR_NOINNV){
            adv_no_innv_seg.push_back(it_seg_to->first->parrent);
        }
    }
    for(int j=0;j<sensor.frame_old->seg_init->size();j++){
        bool found = false;
        for(int i=0;i<sensor.frame_old->seg_ext->size();i++){
            if(sensor.frame_old->seg_ext->at(i)->parrent == sensor.frame_old->seg_init->at(j)){
                found = true;
                break;
            }
        }
        if(!found){
            adv_no_innv_seg.push_back(sensor.frame_old->seg_init->at(j));
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init(State rob_x){
    pos_init = false;
    S.release(); S_R_bar.release(); P.release();;
    Oi.clear();
    v_static = 0;
    w_static = 0;
    input_noise[0] = 1;//v in v noise
    input_noise[1] = 0.1;//w in v noise
    input_noise[2] = 0.1;//v in w noise
    input_noise[3] = 1;//w in w noise
    input_noise[4] = 0;//100;//dt noise
    obj_noise_x = 0.01;
    obj_noise_y = 0.01;
    obj_noise_phi = 0.0001;

    S_R_bar.push_back(0.);      S_R_bar.push_back(0.);      S_R_bar.push_back(0.);
    S      .push_back(rob_x.x); S      .push_back(rob_x.y); S      .push_back(rob_x.angle);
    P = Mat::zeros(rob_param, rob_param, CV_64F);

    S_R_bar_old.push_back(0.);      S_R_bar_old.push_back(0.);      S_R_bar_old.push_back(0.);
    S_old      .push_back(rob_x.x); S_old      .push_back(rob_x.y); S_old      .push_back(rob_x.angle);
    P_old = Mat::zeros(rob_param, rob_param, CV_64F);

    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
    //cout<<"M="<<endl<<" "<<M<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::prediction(KControl u){
    //cout<<"prediction:"<<endl;

    Mat Gt (Mat::zeros(P.rows, P.cols, CV_64F));
    Mat Gt_R; Gt_R = Gt.rowRange(0, rob_param).colRange(0, rob_param); Mat(Mat::eye(3,3,CV_64F)).copyTo(Gt_R);
    Mat Vt_R = Mat::eye(rob_param, input_param, CV_64F);
    Mat Q(P.rows,P.cols,CV_64F, 0.);
    RState  rob_f0(S);

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
    double sign;
    if(fabs(u.v) > fabs(v_static)){ sign = sgn(u.v); }
    else{                           sign = sgn(v_static);}
    double v_noise = sign * fmax(fabs(u.v), fabs(v_static));
    v_static = (v_static + u.v) / 2.0;
    if(fabs(u.w) > fabs(w_static)){ sign = sgn(u.w); }
    else{                           sign = sgn(w_static);}
    double w_noise = sign * fmax(fabs(u.w), fabs(w_static));
    w_static = (w_static + u.w) / 2.0;

    //////INPUT NOISE MATRIX----
    Mat M = Mat::zeros(input_param , input_param, CV_64F);
    M.row(0).col(0) = input_noise[0]*sqr(v_noise) + input_noise[1] * sqr(w_noise);
    M.row(1).col(1) = input_noise[2]*sqr(v_noise) + input_noise[3] * sqr(w_noise);
    M.row(2).col(2) = input_noise[4]*sqr(u.dt);

    Mat(Vt_R * M * Vt_R.t()).copyTo(Q.rowRange(0,rob_param).colRange(0,rob_param));
    ///----INPUT NOISE MATRIX

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

    Mat((Gt * P * Gt.t()) + Q).copyTo(P);///COVARIANCE PREDICTED

    //cout<<"Vt_R="<<endl<<" "<<Vt_R<<endl<<endl;
    //cout<<"Gt="  <<endl<<" "<<Gt  <<endl<<endl;
    //cout<<"M="   <<endl<<" "<<M   <<endl<<endl;
    //cout<<"Q="   <<endl<<" "<<Q   <<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
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

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init_Oi(ObjectDataPtr obj, xy obj_com_bar_f1){
    RState rob_bar_f0(S);
    if(Oi.count(obj) == 0){ return; }
    int i_min = Oi[obj].i_min;

    S.row(i_min + 0) = rob_bar_f0.xx + obj_com_bar_f1.x * cos(rob_bar_f0.xphi) - obj_com_bar_f1.y * sin(rob_bar_f0.xphi);
    S.row(i_min + 1) = rob_bar_f0.xy + obj_com_bar_f1.y * cos(rob_bar_f0.xphi) + obj_com_bar_f1.x * sin(rob_bar_f0.xphi);
    S.row(i_min + 2) = rob_bar_f0.xphi ;
}


void KalmanSLDM::update_Oi_with_Oj(ObjectDataPtr seg, ObjectDataPtr seg_obs){// TODO full object observation (merging of other tracked object)
    //cout<<"innovation:"<<endl;

    ///PREDICTED OBSERVATION----
    Mat h_bar(obj_param, 1, CV_64F, 0.0);
    Oi[seg].S_O.copyTo(h_bar);
    ///----PREDICTED OBSERVATION

    ///PREDICTED OBSERVATION JACOBIAN AND NOISE----
    Mat H_low(obj_param, rob_param + obj_param, CV_64F, 0.0);
    Mat(Mat::eye(obj_param, obj_param, CV_64F)).copyTo(H_low.colRange(rob_param, rob_param + obj_param));
    Mat H = H_low * Fxi(seg);

    Mat Q;Mat(Oi[seg_obs].P_OO).copyTo(Q);
    ///----PREDICTED OBSERVATION JACOBIAN AND NOISE

    Mat h_hat(obj_param, 1, CV_64F, 0.0);
    Oi[seg_obs].S_O.copyTo(h_hat); ///OBSERVATION

    Mat Kt = P * H.t() * ( H * P * H.t() + Q ).inv();                  ///KALMAN GAIN
    Mat(S + Kt * ( h_hat - h_bar )).copyTo(S);                         ///STATE UPDATE
    Mat((cv::Mat::eye(P.rows, P.cols, CV_64F) - Kt * H) * P).copyTo(P);///COVARIANCE UPDATE

    //cout<<"h_hat_f1=" <<endl<<" "<<h_hat_f1                        <<endl<<endl;
    //cout<<"h_bar_f1=" <<endl<<" "<<h_bar_f1                        <<endl<<endl;
    //cout<<"h_diff_f1="<<endl<<" "<<Mat(h_hat_f1 - h_bar_f1)        <<endl<<endl;
    //cout<<"Kt="       <<endl<<" "<<Kt                              <<endl<<endl;
    //cout<<"Kt*dh"     <<endl<<" "<<Mat(Kt * (h_hat_f1 - h_bar_f1 ))<<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
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
//    cout<<"h_diff_f1="<<endl<<" "<<Mat(h_hat_f1 - h_bar_f1)        <<endl<<endl;
    //cout<<"Kt="       <<endl<<" "<<Kt                              <<endl<<endl;
    //cout<<"Kt*dh"     <<endl<<" "<<Mat(Kt * (h_hat_f1 - h_bar_f1 ))<<endl<<endl;
    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool KalmanSLDM::add_obj(ObjectDataPtr seg, KObjZ kObjZ){
//    cout<<"adding object with id"<<endl;

    if(Oi.count(seg) > 0){
//        cout<<"warning, adding an allready inserted object"<<endl;
        return false;
    }

    RState  rob_bar_f0(S);
    OiState obj_zhat_f1;
    obj_zhat_f1.xx = kObjZ.pos.x; obj_zhat_f1.xy = kObjZ.pos.y; obj_zhat_f1.xphi = kObjZ.phi;

    //adding the object state parameters
    S.push_back(rob_bar_f0.xx   + obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi));
    S.push_back(rob_bar_f0.xy   + obj_zhat_f1.xy * cos(rob_bar_f0.xphi) + obj_zhat_f1.xx * sin(rob_bar_f0.xphi));
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
    Gt_hinv_R.row(0).col(0) = 1.; Gt_hinv_R.row(1).col(1) = 1.; Gt_hinv_R.row(2).col(2) = 1.;

    Gt_hinv_R.row(0).col(2) = - obj_zhat_f1.xy * cos(rob_bar_f0.xphi) - obj_zhat_f1.xx * sin(rob_bar_f0.xphi);
    Gt_hinv_R.row(1).col(2) =   obj_zhat_f1.xx * cos(rob_bar_f0.xphi) - obj_zhat_f1.xy * sin(rob_bar_f0.xphi);

//    Mat Gt_hinv_Z (obj_param, z_param, CV_64F, 0.);
//    Gt_hinv_Z.row(0).col(0) = cos(rob_bar_f0.xphi); Gt_hinv_Z.row(0).col(1) = - sin(rob_bar_f0.xphi);
//    Gt_hinv_Z.row(1).col(0) = sin(rob_bar_f0.xphi); Gt_hinv_Z.row(1).col(1) =   cos(rob_bar_f0.xphi);
//    Gt_hinv_Z.row(2).col(2) = 1.;

//    Mat(Gt_hinv_R * P_RR * Gt_hinv_R.t()  + Gt_hinv_Z * Mat(kObjZ.Q)  * Gt_hinv_Z.t()).copyTo(Oi[seg].P_OO);  ???
    Mat P_OO_V(9, 9, CV_64F, 0.);
    P_OO_V.row(0).col(0) = 1;//initial covariance of x,v,a TODO....for x you should add the Q stuff
    P_OO_V.row(1).col(1) = 1;
    P_OO_V.row(2).col(2) = 1;
    P_OO_V.row(3).col(3) = 1;
    P_OO_V.row(4).col(4) = 1;
    P_OO_V.row(5).col(5) = 1;
    P_OO_V.row(6).col(6) = 1;
    P_OO_V.row(7).col(7) = 1;
//    P_OO_V.row(8).col(8) = 1;
    P_OO_V.copyTo(Oi[seg].P_OO/*.rowRange(6,9).colRange(6,9)*/);

    Mat(Gt_hinv_R * P.rowRange(0, rob_param).colRange(0, P.cols - obj_param)).copyTo(P_ORi);
    Mat(P_ORi.t()).copyTo(P_ROi);
    ///OBJECT COVARIANCE INITIALIZATION----


    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
//    cout<<"P="<<endl<<" "<<P<<endl<<endl;
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool KalmanSLDM::rmv_obj(ObjectDataPtr seg){
    //cout<<"removing object with id"<<test_id<<endl;
    if(Oi.count(seg) == 0){
//        cout<<"attempting to remove inexisting key"<<endl;
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
