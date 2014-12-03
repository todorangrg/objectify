#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"

using namespace cv;
using namespace std;
//TODO include displacement of the sensor in the model

//TODO : correct and tf all points after the update stuff


//class NeighDatasimple{
//public:
//    SegmentDataPtr neigh;
//    double         prob_fwd;
//    double         prob_rev;
//    NeighDatasimple(SegmentDataPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev){}
//};

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::advance(InputData &input, bool advance){
    if(advance){
        input.u.dt = (input.time_stamp - time_stamp).toSec();
        if(input.u.v != input.u.v){ input.u.v = 0; }// NaN stuff
        if(input.u.w != input.u.w){ input.u.w = 0; }

        update_sub_mat();
        S_R_bar_old = Mat(S_R_bar.rows, S_R_bar.cols, CV_64F); S_R_bar.copyTo(S_R_bar_old);//storing a copy of actual state matrices
        S_old       = Mat(S      .rows, S      .cols, CV_64F); S      .copyTo(S_old);
        P_old       = Mat(P      .rows, P      .cols, CV_64F); P      .copyTo(P_old);
        Oi_old      = Oi;                                                                  //storing a copy of actual object map
        if(Oi.size() > 0){                                                                 //storing a copy of actual kalman seg_init
            SegCopy(seg_init, seg_init_old);
        }
    }
    else{
        S_R_bar = Mat(S_R_bar_old.rows, S_R_bar_old.cols, CV_64F); S_R_bar_old.copyTo(S_R_bar);
        S       = Mat(S_old      .rows, S_old      .cols, CV_64F); S_old      .copyTo(S);
        P       = Mat(P_old      .rows, P_old      .cols, CV_64F); P_old      .copyTo(P);
        Oi      = Oi_old;
        if(Oi.size() > 0){
            SegCopy(seg_init_old, seg_init);
        }
    }
    update_sub_mat();
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::run(InputData &input,
                     std::map <SegmentDataExtPtr, std::vector<NeighDataExt > >& neigh_data_oe,
                     std::map <SegmentDataExtPtr, std::vector<NeighDataExt > >& neigh_data_ne,
                     std::map <SegmentDataPtr   , std::vector<NeighDataInit> >& neigh_data_oi){

    if(!input.seg_init){ return; }

    seg_init = SegmentDataPtrVectorPtr(new SegmentDataPtrVector);

    int id_plus=0;


    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        oi->first->solved = false;
    }
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
        if(oi->first->solved == true){ continue; }

        std::vector<CorrInput>     list_comm; list_comm.reserve(neigh_data_oe.size() * neigh_data_ne.size());
        std::vector<ObjectDataPtr>    o_comm;    o_comm.reserve(Oi.size());

        oi->first->solved = true;//flag object as extracted
        o_comm.push_back(oi->first);//append the object in the common list

        extract_common_pairs(o_comm, list_comm, neigh_data_oe, neigh_data_ne);//populate o_comm and list_comm according to found object-segment links

        //////////////TODO analysis of merge/split (here you need a list of a vector of pairs for each resolved connect graph)
        //////////////TODO duplicate objects that are being splitted

        //erasing duplicate of objects that are being merged
        for(std::vector<ObjectDataPtr>::iterator o_comm_it = o_comm.begin() + 1; o_comm_it != o_comm.end(); o_comm_it++){
            rmv_obj(*o_comm_it);
        }

        //update the stuff
        ///TODO ENCHANTED----

        //computing avg miu and sigma
        KObjZ upd_data;
        compute_avg_miu_sigma(list_comm, upd_data);

        init_Oi  (oi->first, xy(0,0));
        update_Oi(oi->first, upd_data);


        for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){//here you need to insert INIT
            bool found = false;
            bool p_tf = false;
            SegmentDataPtr sss;
            bool seg_unique = true;
            for(std::vector<CorrInput>::iterator entry_search = list_comm.begin(); entry_search != list_comm.end(); entry_search++){
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
        cv::RotatedRect rect = cov2rect(cv::Matx22d(oi->second.P_OO.rowRange(0,2).colRange(0,2)),xy(0,0));
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

bool KalmanSLDM::compute_avg_miu_sigma(std::vector<CorrInput> & list_comm, KObjZ &avg){
    Mat P_oi_avg(z_param, z_param, CV_64F, 0.);
    Mat S_oi_avg(z_param,       1, CV_64F, 0.);
    if(list_comm.size() == 0){ return false; }

    for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){
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

        if(list_comm.size() > 1){
            P_oi_avg = P_oi_avg + sig.inv();
            S_oi_avg = S_oi_avg + sig.inv() * miu;
        }
        else{
            P_oi_avg = sig;
            S_oi_avg = miu;
        }
    }
    if(list_comm.size() > 1){
        P_oi_avg = P_oi_avg.inv();// sigma = sum( sigma_k(-1))(-1)
        S_oi_avg = P_oi_avg * S_oi_avg;// miu = sigma * sum(sigma_k(-1) * miu_k)
    }

    avg.pos.x   = S_oi_avg.at<double>(0);
    avg.pos.y   = S_oi_avg.at<double>(1);
    avg.phi     = S_oi_avg.at<double>(2);
    avg.Q       = cv::Matx33d(P_oi_avg);

    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::extract_common_pairs(std::vector<ObjectDataPtr>                               & o_comm,
                                      std::vector<CorrInput>                                   & list_comm,
                                      std::map <SegmentDataExtPtr, std::vector<NeighDataExt > >& neigh_data_oe,
                                      std::map <SegmentDataExtPtr, std::vector<NeighDataExt > >& neigh_data_ne){

    bool research = false;
    do{
        research = false;
        // search in seg_ext (t-1) neighbouring list
        for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator s_o = neigh_data_oe.begin(); s_o != neigh_data_oe.end(); s_o++){
            if(s_o->first->solved){ continue; }//if extracted already continue

            //search if the object of s_o is being extracted now
            for(std::vector<ObjectDataPtr>::iterator o_comm_it = o_comm.begin(); o_comm_it != o_comm.end(); o_comm_it++){
                if(s_o->first->getObj() != *o_comm_it){ continue; }//if not extracted now continue

                s_o->first->solved = true; s_o->first->getParrent()->solved = true; //flag seg_ext(t-1) and seg_init(t-1) from pair as extracted

                //search in all the seg_ext(t-1) neighbours
                for(std::vector<NeighDataExt>::iterator s_o_neigh = s_o->second.begin(); s_o_neigh != s_o->second.end(); s_o_neigh++){
                    if(s_o_neigh->neigh->solved){ continue; }//if extracted already continue

                    s_o_neigh->neigh->solved = true; s_o_neigh->neigh->getParrent()->solved = true; //flag seg_ext(t) and seg_init(t) from pair as extracted

                    list_comm.push_back(CorrInput(s_o->first, s_o_neigh->neigh,0,0));//append the pair in the common list

                    //search in seg_ext(t) for the appended segments and append other possible links that have objects not yet in o_comm
                    for(std::vector<NeighDataExt>::iterator s_n = neigh_data_ne[s_o_neigh->neigh].begin(); s_n != neigh_data_ne[s_o_neigh->neigh].end(); s_n++){
                        if(s_n->neigh->solved){ continue; }//if extracted already continue

                        //search for new objects, that are not yet in o_comm
                        bool found = false;
                        for(std::vector<ObjectDataPtr>::iterator o_comm_it2 = o_comm.begin(); o_comm_it2 != o_comm.end(); o_comm_it2++){
                            if(s_n->neigh->getObj()){
                                if(*o_comm_it2 == s_n->neigh->getObj()){
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if(found){ continue; }//if not a new object in extraction continue

                        s_n->neigh->getObj()->solved = true;//flag object as extracted
                        s_n->neigh->solved = true; s_n->neigh->getParrent()->solved = true; //flag seg_ext(t-1) and seg_init(t-1) from pair as extracted

                        o_comm.push_back(s_n->neigh->getObj());//append the object in the common list
                        list_comm.push_back(CorrInput(s_n->neigh, s_o_neigh->neigh,0,0));//append the pair in the common list

                        research = true;//if new object inserted, rerun extraction, some segments might have been previously skipped
                    }
                }
            }
        }
    }while(research);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
