#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;
//TODO include displacement of the sensor in the model






class NeighDatasimple{
public:
    SegmentDataPtr neigh;
    double         prob_fwd;
    double         prob_rev;
    NeighDatasimple(SegmentDataPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev){}
};



void KalmanSLDM::advance(InputData &input, bool advance){//TODO deal with "this only if "timeout done / covariance of reprediction too big" : check it in the old sensor frame, in the no_innovation flag and move them in new frame
    if(advance){
//        if(Oi.size() > 0){
//            seg_init = seg_init_plus;
//        }

        //TODO : correct and tf all points after the update stuff
        update_sub_mat();
        input.u.dt = (input.time_stamp - time_stamp).toSec();

        S_R_bar_old = Mat(S_R_bar.rows,S_R_bar.cols,CV_64F); S_R_bar.copyTo(S_R_bar_old);
        S_old = Mat(S.rows,S.cols,CV_64F); S.copyTo(S_old);
        P_old = Mat(P.rows,P.cols,CV_64F); P.copyTo(P_old);
        Oi_old = Oi;
        if(Oi.size() > 0){
            //seg_init_old = seg_init;
            //seg_init = seg_init_new;
            seg_init_old = SegmentDataPtrVectorPtr ( new SegmentDataPtrVector);
            for(SegmentDataPtrVectorIter ss = seg_init->begin(); ss != seg_init->end(); ss++){
                seg_init_old->push_back(SegmentDataPtr(new  SegmentData(ss)));
                for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
                    seg_init_old->back()->p.push_back(*pp);
                }
            }
        }
    }
    else{
        S_R_bar = Mat(S_R_bar_old.rows,S_R_bar_old.cols,CV_64F); S_R_bar_old.copyTo(S_R_bar);
        S = Mat(S_old.rows,S_old.cols,CV_64F); S_old.copyTo(S);
        P = Mat(P_old.rows,P_old.cols,CV_64F); P_old.copyTo(P);
        Oi = Oi_old;
        if(Oi.size() > 0){
            //seg_init = seg_init_old;
            seg_init = SegmentDataPtrVectorPtr ( new SegmentDataPtrVector);
            for(SegmentDataPtrVectorIter ss = seg_init_old->begin(); ss != seg_init_old->end(); ss++){
                seg_init->push_back(SegmentDataPtr(new  SegmentData(ss)));
                for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
                    seg_init->back()->p.push_back(*pp);
                }
            }
        }

    }
    update_sub_mat();
}

cv::RotatedRect cov2rectt(cv::Matx<double, 2, 2> _C,xy _center) {
    cv::RotatedRect ellipse;
    cv::Mat_<double> eigval, eigvec;
    cv::eigen(_C, eigval, eigvec);

    /// Exercise4
    bool index_x;
    ellipse.center = _center;
    if(_C(0,0)>_C(1,1)){
        index_x=0;
    }
    else{
        index_x=1;
    }
    ellipse.size.height=sqrt(fabs(eigval(0,!index_x)))*2.4477;//y
    ellipse.size.width=sqrt(fabs(eigval(0,index_x)))*2.4477;//x
    if((eigval(0,index_x)!=0)&&(eigval(0,!index_x)!=0))
        ellipse.angle=atan2(eigvec(index_x,1),eigvec(index_x,0))*(180/M_PI);

    return ellipse;
}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::run(InputData &input,
                     std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >& neigh_data_o,
                     std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >& neigh_data_n,
                     std::map <SegmentDataPtr, std::vector<NeighDataInit> >& neigh_data_oi){

    if(!input.seg_init){
        return;
    }

    seg_init = SegmentDataPtrVectorPtr(new SegmentDataPtrVector);


    int id_plus=0;


    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        oi->first->solved = false;
    }
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        std::vector<CorrInput> mod_list;
        mod_list.reserve(neigh_data_o.size() * neigh_data_n.size());
        if(oi->first->solved == true){
            continue;
        }
        std::vector<ObjectDataPtr> o_mod;
        o_mod.reserve(Oi.size());
        o_mod.push_back(oi->first);
        oi->first->solved = true;///flag object as solved
        for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator s_o = neigh_data_o.begin(); s_o != neigh_data_o.end(); s_o++){// search on seg_ext (t-1) neighbouring list
            if(s_o->first->solved){
                continue;
            }
            for(std::vector<ObjectDataPtr>::iterator o_mod_it = o_mod.begin(); o_mod_it != o_mod.end(); o_mod_it++){//compare object of their key with "dealing with now" objects
                if(s_o->first->getObj() == *o_mod_it){//if the same
                    s_o->first->solved = true;///flag t-1 seg_ext from pair as solved
                    s_o->first->getParrent()->solved = true;
                    for(std::vector<NeighDataExt>::iterator s_ext_t = s_o->second.begin(); s_ext_t != s_o->second.end(); s_ext_t++){//look at all those segment neighbours
                        if(s_ext_t->neigh->solved){
                            continue;
                        }
                        mod_list.push_back(CorrInput(s_o->first, s_ext_t->neigh,0,0));//append the pairs
                        s_ext_t->neigh->solved = true;///flag t-1 seg_ext from pair as solved (incl parrent)
                        s_ext_t->neigh->getParrent()->solved = true;
                        for(std::vector<NeighDataExt>::iterator s_n = neigh_data_n[s_ext_t->neigh].begin(); s_n != neigh_data_n[s_ext_t->neigh].end(); s_n++){//look in seg_ext(t) for the appended segments and append other possible links that have other than "dealing with now" objects
                            if(s_n->neigh->solved){
                                continue;
                            }
                            bool found = false;
                            for(std::vector<ObjectDataPtr>::iterator o_mod_it2 = o_mod.begin(); o_mod_it2 != o_mod.end(); o_mod_it2++){
                                if(s_n->neigh->getObj()){
                                    if(*o_mod_it2 == s_n->neigh->getObj()){
                                        found = true;
                                        break;
                                    }
                                }
                            }
                            if(!found){
                                o_mod.push_back(s_n->neigh->getObj());
                                s_n->neigh->getObj()->solved = true;
                                mod_list.push_back(CorrInput(s_n->neigh, s_ext_t->neigh,0,0));
                                s_n->neigh->solved = true;///flag t seg_ext from pair as solved (incl parrent)
                                s_n->neigh->getParrent()->solved = true;
                            }
                        }
                    }
                }
            }
        }
        //here we have the list of all pairs that are potential to get merged and the list of "dealing with now" objects

        //////////////analysis of merge/split (here you need a list of a vector of pairs for each resolved connect graph)




        ///duplicate objects that are being splitted
        //erase   objects that are being merged
        for(std::vector<ObjectDataPtr>::iterator o_mod_it = o_mod.begin() + 1; o_mod_it != o_mod.end(); o_mod_it++){
            rmv_obj(*o_mod_it);
        }

        //update the stuff
        ///TODO ENCHANTED----
        Mat P_oi_avg(z_param, z_param, CV_64F, 0.);
        Mat S_oi_avg(z_param,1,CV_64F,0.);
        for(std::vector<CorrInput>::iterator entry = mod_list.begin(); entry != mod_list.end(); entry++){//compute avg miu and sigma
            TFdata tf;
            for(std::vector<TFdata>::iterator tf_it = entry->frame_old->conv->tf->begin(); tf_it != entry->frame_old->conv->tf->end(); tf_it++){//extract the needed tf
                if(tf_it->seg == entry->frame_new){
                    tf = *tf_it;
                    break;
                }
            }
            xy     com_new = tf.tf.xy_mean;//new com in rob_bar frame after motion prediction
            double phi_new = tf.tf.ang_mean;

            Mat miu(z_param,1,CV_64F,0.);

            miu.row(0) = com_new.x;
            miu.row(1) = com_new.y;
            miu.row(2) = phi_new;
            Mat sig(z_param, z_param, CV_64F, 0.);
            sig.row(0).col(0) = tf.tf.xy_cov(0,0);
            sig.row(0).col(1) = tf.tf.xy_cov(0,1);
            sig.row(1).col(0) = tf.tf.xy_cov(1,0);
            sig.row(1).col(1) = tf.tf.xy_cov(1,1);
            sig.row(2).col(2) = tf.tf.ang_cov;//all this is in rob_bar frame

            if(mod_list.size() > 1){
                P_oi_avg = P_oi_avg + sig.inv();
                S_oi_avg = S_oi_avg + sig.inv() * miu;
            }
            else{
                P_oi_avg = sig;
                S_oi_avg = miu;
            }
        }
        if(mod_list.size() > 1){
            P_oi_avg = P_oi_avg.inv();// sigma = sum( sigma_k(-1))(-1)
            S_oi_avg = P_oi_avg * S_oi_avg;// miu = sigma * sum(sigma_k(-1) * miu_k)
        }

        if(mod_list.size() > 0){
            //UPDATEUPDATEUPDATE!!!!!!!!!

            KObjZ val;
            val.pos.x   = S_oi_avg.at<double>(0);
            val.pos.y   = S_oi_avg.at<double>(1);
            val.phi     = S_oi_avg.at<double>(2);
            val.Q(0,0)  = P_oi_avg.at<double>(0,0);
            val.Q(0,1)  = P_oi_avg.at<double>(0,1);
            val.Q(1,0)  = P_oi_avg.at<double>(1,0);
            val.Q(1,1)  = P_oi_avg.at<double>(1,1);
            val.Q(2,2)  = P_oi_avg.at<double>(2,2);
            init_Oi(oi->first, xy(0,0));
            update_Oi(oi->first, val);
        }


        for(std::vector<CorrInput>::iterator entry = mod_list.begin(); entry != mod_list.end(); entry++){//here you need to insert INIT
            bool found = false;
            bool p_tf = false;
            SegmentDataPtr sss;
            bool seg_unique = true;
            for(std::vector<CorrInput>::iterator entry_search = mod_list.begin(); entry_search != mod_list.end(); entry_search++){
                if((entry_search != entry)&&((entry_search->frame_old->getParrent() == entry->frame_old->getParrent())||
                   (entry_search->frame_new->getParrent() == entry->frame_new->getParrent()))){
                    seg_unique = false;
                    break;
                }
            }
            if(/*(found)&&*/(seg_unique)&&(entry->frame_old->getParrent()->getLen() > entry->frame_new->getParrent()->getLen() + 0.01)){
                sss = entry->frame_old->getParrent();
                p_tf = true;
            }
            else{
                sss = entry->frame_new->getParrent();
            }
            for(SegmentDataPtrVectorIter it_seg_ins = seg_init->begin(); it_seg_ins != seg_init->end(); it_seg_ins++){
                if(*it_seg_ins == sss){
                    found = true;
                    break;
                }
            }
            if(!found){
                seg_init->push_back(SegmentDataPtr(new  SegmentData(oi->first,id_plus)));
                if(p_tf){
                    //for(PointDataVectorIter pp = sss->p_tf.begin(); pp != sss->p_tf.end(); pp++){
                    for(PointDataVectorIter pp = sss->p.begin(); pp != sss->p.end(); pp++){
                        seg_init->back()->p.push_back(to_polar(mat_mult(entry->frame_old->conv->tf->front().tf.T,to_xy(*pp))));
                    }
                }
                else{
                    for(PointDataVectorIter pp = sss->p.begin(); pp != sss->p.end(); pp++){
                        seg_init->back()->p.push_back(*pp);
                    }
                }
            }
            id_plus++;
        }
        ///if 1 to 1 in INIT, choose to keep the longer one
        ///----TODO ENCHANTED

    }
    for(SegmentDataPtrVectorIter s_ne =  input.seg_init->begin(); s_ne != input.seg_init->end(); s_ne ++){
        if(((*s_ne)->solved)){
            continue;
        }
        bool found = false;
        for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator s_oi = neigh_data_oi.begin(); s_oi != neigh_data_oi.end(); s_oi++){
            for(std::vector<NeighDataInit>::iterator s_ni = s_oi->second.begin(); s_ni != s_oi->second.end(); s_ni++){
                if(*s_ne == s_ni->neigh){
                    found = true;
                    break;
                }
            }
        }
        if(!found){
            seg_init->push_back(SegmentDataPtr(new  SegmentData(ObjectDataPtr(new ObjectData(id_plus  /* here wrong, need counter */)),id_plus)));
            for(PointDataVectorIter pp = (*s_ne)->p.begin(); pp != (*s_ne)->p.end(); pp++){
                seg_init->back()->p.push_back(*pp);
            }
            id_plus++;

            KObjZ dummy;
            init_Oi(seg_init->back()->getObj(),xy(0,0));
            update_Oi(seg_init->back()->getObj(),dummy);
            //UPDATEUPDATEUPDATE!!!!!!!!!
        }
    }
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        cv::RotatedRect rect = cov2rectt(cv::Matx22d(oi->second.P_OO.rowRange(0,2).colRange(0,2)),xy(0,0));
        if(rect.size.height * rect.size.width > 10){
            ObjectDataPtr rmv = oi->first;
            oi--;
            rmv_obj(rmv);
        }
    }
    for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator s_oi = neigh_data_oi.begin(); s_oi != neigh_data_oi.end(); s_oi++){
        if((s_oi->second.size() > 0)||(Oi.count(s_oi->first->getObj()) == 0)){
            continue;
        }
        seg_init->push_back(SegmentDataPtr(new  SegmentData(s_oi->first->getObj(),id_plus)));

        //for(PointDataVectorIter pp = s_oi->first->p_tf.begin(); pp != s_oi->first->p_tf.end(); pp++){
        for(PointDataVectorIter pp = s_oi->first->p.begin(); pp != s_oi->first->p.end(); pp++){
            seg_init->back()->p.push_back(*pp);
        }
        id_plus++;
    }
    //look for not flagged seg_init (t)
    // ->that have neighbours in the neigh_list and erase them (dont keep the point clouds for the next frame)
    // ->that have no neighbours => init them as new objects
    //DONE!
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::init(RState rob_x){
    seg_init_plus = SegmentDataPtrVectorPtr(new SegmentDataPtrVector());
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
    S      .push_back(rob_x.xx); S      .push_back(rob_x.xy); S      .push_back(rob_x.xphi);
    P = Mat::zeros(rob_param, rob_param, CV_64F);

    S_R_bar_old.push_back(0.);      S_R_bar_old.push_back(0.);      S_R_bar_old.push_back(0.);
    S_old      .push_back(rob_x.xx); S_old      .push_back(rob_x.xy); S_old      .push_back(rob_x.xphi);
    P_old = Mat::zeros(rob_param, rob_param, CV_64F);

    //cout<<"S="<<endl<<" "<<S<<endl<<endl;
    //cout<<"P="<<endl<<" "<<P<<endl<<endl;
    //cout<<"M="<<endl<<" "<<M<<endl<<endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::prediction(SegmentDataPtrVectorPtr &input, KInp u){
    //cout<<"prediction:"<<endl;

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
//    cout<<"P="<<endl<<" "<<P<<endl<<endl;
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
