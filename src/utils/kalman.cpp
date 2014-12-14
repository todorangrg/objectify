#include "utils/kalman.h"
#include "data_processing/correlation.h"
#include "visual/plot.h"
#include "math.h"

using namespace cv;
using namespace std;

//TODO : angle stuff

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::advance(InputData &input, bool advance){
    if(advance){
        input.u.dt = (input.time_stamp - time_stamp).toSec();
        if(input.u.v != input.u.v){ input.u.v = 0; }// NaN stuff
        if(input.u.w != input.u.w){ input.u.w = 0; }

        update_sub_mat();
        S_old       = Mat(S.rows, S.cols, CV_64F); S.copyTo(S_old);//storing a copy of actual state matrices
        P_old       = Mat(P.rows, P.cols, CV_64F); P.copyTo(P_old);
        Oi_old      = Oi;                                          //storing a copy of actual object map
        if(Oi.size() > 0){                                         //storing a copy of actual kalman seg_init
            SegCopy(seg_init, seg_init_old);
        }
    }
    else{
        S       = Mat(S_old.rows, S_old.cols, CV_64F); S_old.copyTo(S);
        P       = Mat(P_old.rows, P_old.cols, CV_64F); P_old.copyTo(P);
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
                     std::map <SegmentDataPtr   , std::vector<NeighDataInit> >& neigh_data_oi,
                     std::map <SegmentDataPtr   , std::vector<NeighDataInit> >& neigh_data_ni){

    if(input.seg_init){ seg_init = SegmentDataPtrVectorPtr(new SegmentDataPtrVector); }

    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){ oi->first->solved = false; }
    bool done = false;
    while(!done){
        done = true;
        for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
            if(oi->first->solved == true){ continue; }

            std::vector<CorrInput>     list_comm; list_comm.reserve(neigh_data_oe.size() * neigh_data_ne.size());
            std::vector<ObjectDataPtr>    o_comm;    o_comm.reserve(Oi.size());

            oi->first->solved = true;   //flag object as extracted
            o_comm.push_back(oi->first);//append the object in the common list

            extract_common_pairs(o_comm, list_comm, neigh_data_oe, neigh_data_ne);//populate o_comm and list_comm according to found object-segment links

            //////////////TODO analysis of merge/split (here you need a list of a vector of pairs for each resolved connect graph)
            //////////////TODO duplicate objects that are being splitted

            //erasing duplicate of objects that are being merged
            for(std::vector<ObjectDataPtr>::iterator o_comm_it = o_comm.begin() + 1; o_comm_it != o_comm.end(); o_comm_it++){
                rmv_obj(*o_comm_it); done = false;
            }

            KObjZ upd_data;//update the stuff
            if(compute_avg_miu_sigma(list_comm, upd_data)){
                init_Oi  (oi->first, xy(0,0), input.u.dt);
                update_Oi(oi->first, upd_data);
            }

            propag_extr_p_clouds(list_comm, oi);
            if(!done) { break; }
        }
    }

    add_new_obj(input.seg_init, neigh_data_ni);

    remove_lost_obj();

    propagate_no_update_obj(neigh_data_oi, neigh_data_ni);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::propagate_no_update_obj(std::map <SegmentDataPtr, std::vector<NeighDataInit> >& neigh_data_oi, std::map <SegmentDataPtr  , std::vector<NeighDataInit> > & neigh_data_ni){
    //search in seg_init (t-1)
    for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator s_oi = neigh_data_oi.begin(); s_oi != neigh_data_oi.end(); s_oi++){
        if((s_oi->first->solved)||(Oi.count(s_oi->first->getObj()) == 0)){ continue; }//if segment has neighbours or object was erased discard it

        bool ask_continue = false;
        for(std::vector<NeighDataInit>::iterator it_neigh = s_oi->second.begin(); it_neigh != s_oi->second.end(); it_neigh++){
            if(!it_neigh->has_tf){
                for(std::vector<NeighDataInit>::iterator it_neigh_n = neigh_data_ni[it_neigh->neigh].begin(); it_neigh_n != neigh_data_ni[it_neigh->neigh].end(); it_neigh_n++){
                    if(it_neigh_n->has_tf){
                        ask_continue = true;
                        break;
                    }
                }
            }
        }
        if(ask_continue){ continue; }
        //else propagate it
        seg_init->push_back(SegmentDataPtr(new  SegmentData(s_oi->first)));
        for(PointDataVectorIter pp = s_oi->first->p.begin(); pp != s_oi->first->p.end(); pp++){
            seg_init->back()->p.push_back(*pp);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::remove_lost_obj(){
    bool no_remove = true;
    do{
        no_remove = true;
        for(std::map<ObjectDataPtr, ObjMat>::iterator oi = Oi.begin(); oi != Oi.end(); oi++){
            cv::RotatedRect rect = cov2rect(cv::Matx22d(oi->second.P_OO.rowRange(0,2).colRange(0,2)),xy(0,0));
            // if x_var * y_var > threshold => erase object
            if(rect.size.height * rect.size.width > sqr(obj_timeout)){//TODO make it parameter
                ObjectDataPtr rmv = oi->first;
                rmv_obj(rmv);
                no_remove = false;
                break;
            }
        }
    }while(!no_remove);// this while because map doesn't give back an iterator after erasing :|
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::add_new_obj(SegmentDataPtrVectorPtr & inp_seg_init, std::map <SegmentDataPtr, std::vector<NeighDataInit> >& neigh_data_ni){
    if(inp_seg_init){
        //search in all new init segments
        for(SegmentDataPtrVectorIter s_ne =  inp_seg_init->begin(); s_ne != inp_seg_init->end(); s_ne ++){
            if(((*s_ne)->solved)||(neigh_data_ni[*s_ne].size() > 0)){ continue; }// if not solved yet ( a.k.a they have no tf )
            bool found = false;
            //search if they have any neighbours
//            for(std::vector<NeighDataInit> ::iterator s_ni = neigh_data_oi[*s_ne].begin(); s_ni != neigh_data_oi[*s_ne].end(); s_ni++){
//                if((*s_ne == s_ni->neigh)&&(!s_ni->has_tf)){ found = true; break; }
//            }
            //if no, initialize as new objects and propagate point clouds
            if(!found){
                seg_init->push_back(SegmentDataPtr(new  SegmentData(ObjectDataPtr(new ObjectData(assign_unique_id())),0, (*s_ne))));
                for(PointDataVectorIter pp = (*s_ne)->p.begin(); pp != (*s_ne)->p.end(); pp++){
                    seg_init->back()->p.push_back(*pp);
                }
                KObjZ dummy;
                update_Oi(seg_init->back()->getObj(),dummy);//actual object update
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void KalmanSLDM::propag_extr_p_clouds(std::vector<CorrInput> & list_comm, std::map<ObjectDataPtr, ObjMat>::iterator oi){

    bool parrent_seg_unique = true;
    //search first time other extended segments have the same init parrent; if yes, take last frame
    for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){
        for(std::vector<CorrInput>::iterator entry_search = list_comm.begin(); entry_search != list_comm.end(); entry_search++){
            if((entry_search != entry)&&((entry_search->frame_old->getParrent() == entry->frame_old->getParrent())||
                                         (entry_search->frame_new->getParrent() == entry->frame_new->getParrent()))){
                parrent_seg_unique = false; break;
            }
        }
    }
    for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){
        //if not and last frame is longer, take that one TODO:: better stuff here
        SegmentDataPtr s_insert; bool p_tf = false;
        if((parrent_seg_unique)&&
                            (entry->frame_old->getParrent()->getLen() > entry->frame_new->getParrent()->getLen() + 0.01)&&
                            (entry->frame_new->conv->tf->front().tf.len / entry->frame_new->conv->p_cd->size() > 0.7)){//HARDCODED
              s_insert = entry->frame_old->getParrent(); p_tf = true; }
        else{ s_insert = entry->frame_new->getParrent();              }
        //search if the segment to be propagated was not propagated already
        bool found = false;
        for(std::vector<CorrInput>::iterator entry_search = list_comm.begin(); entry_search != list_comm.end(); entry_search++){
            if((entry_search != entry)&&((entry->frame_old->getParrent() == s_insert)||(entry_search->frame_new->getParrent() == s_insert))){
                found = true; break;
            }
        }
        if(!found){//if not append it
            seg_init->push_back(SegmentDataPtr(new  SegmentData(oi->first, assign_unique_id(), s_insert)));
            seg_init->back()->solved = true;
            if(p_tf){ seg_init->back()->setCom(mat_mult(entry->frame_old->conv->tf->front().tf.T,seg_init->back()->getCom())); }
            for(PointDataVectorIter pp = s_insert->p.begin(); pp != s_insert->p.end(); pp++){
                ::polar p_in;
                //if from old frame move it according to the tf
                if(p_tf){ p_in = to_polar(mat_mult(entry->frame_old->conv->tf->front().tf.T,to_xy(*pp))); }
                else    { p_in = *pp; }
                seg_init->back()->p.push_back(PointData(p_in));
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool KalmanSLDM::compute_avg_miu_sigma(std::vector<CorrInput> & list_comm, KObjZ &avg){//TODO INCLUDE LEN WEIGHT
    Mat P_oi_avg(z_param - 1, z_param - 1, CV_64F, 0.);
    Mat S_oi_avg(z_param - 1,           1, CV_64F, 0.);
    Gauss ang_avg_cos;
    Gauss ang_avg_sin;

    if(list_comm.size() == 0){ return false; }

    for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){
        TFdata tf;
        for(std::vector<TFdata>::iterator tf_it = entry->frame_old->conv->tf->begin(); tf_it != entry->frame_old->conv->tf->end(); tf_it++){//extract the needed tf
            if(tf_it->seg == entry->frame_new){
                tf = *tf_it;
                break;
            }
        }
        Mat miu(z_param - 1, 1, CV_64F, 0.);
        xy tf_com = mat_mult(tf.tf.T, tf.tf.com);
        tf_com    = tf_com - tf.tf.com;
        miu.row(0) = tf.tf.com_tf.x;//tf_com.x - tf.tf.com.x;
        miu.row(1) = tf.tf.com_tf.y;//tf_com.y - tf.tf.com.y;
        ang_avg_cos.add_w_sample(tf.tf.T(0,0), exp(- tf.tf.Q(2,2) / 2.0) );
        ang_avg_sin.add_w_sample(tf.tf.T(1,0), exp(- tf.tf.Q(2,2) / 2.0) );

        Mat sig(z_param - 1, z_param - 1, CV_64F, 0.);
        Mat Qm(tf.tf.Q);
        Mat(Qm.rowRange(0,2).colRange(0,2)).copyTo(sig);

        if(list_comm.size() > 1){
            P_oi_avg = P_oi_avg + sig.inv(DECOMP_SVD);
            S_oi_avg = S_oi_avg + sig.inv(DECOMP_SVD) * miu;
        }
        else{
            P_oi_avg = sig;
            S_oi_avg = miu;
        }
    }
    if(list_comm.size() > 1){
        P_oi_avg = P_oi_avg.inv(DECOMP_SVD);// sigma =         sum( sigma_k^(-1)        )^(-1)
        S_oi_avg = P_oi_avg * S_oi_avg;     // miu   = sigma * sum( sigma_k^(-1) * miu_k)
    }
    avg.pos.x   = S_oi_avg.at<double>(0);
    avg.pos.y   = S_oi_avg.at<double>(1);
    avg.phi     = atan2(ang_avg_sin.getMean(), ang_avg_cos.getMean());
    avg.Q       = cv::Matx33d(P_oi_avg.at<double>(0,0), P_oi_avg.at<double>(0,1), 0,
                              P_oi_avg.at<double>(1,0), P_oi_avg.at<double>(1,1), 0,
                                                     0,                        0, - 2 * log( sqrt(sqr(ang_avg_cos.getMean()) + sqr(ang_avg_sin.getMean()))));
    cv::Matx33d T(ang_avg_cos.getMean(), - ang_avg_sin.getMean(),         0,
                  ang_avg_sin.getMean(),   ang_avg_cos.getMean(),         0,
                                      0,                       0,         1);
    for(std::vector<CorrInput>::iterator entry = list_comm.begin(); entry != list_comm.end(); entry++){
        TFdata tf;
        for(std::vector<TFdata>::iterator tf_it = entry->frame_old->conv->tf->begin(); tf_it != entry->frame_old->conv->tf->end(); tf_it++){//extract the needed tf
            if(tf_it->seg == entry->frame_new){
                tf = *tf_it;
                xy trans = tf.tf.com + avg.pos - mat_mult(T, tf.tf.com);
                cv::Matx33d Tt(ang_avg_cos.getMean(), - ang_avg_sin.getMean(),   trans.x,
                               ang_avg_sin.getMean(),   ang_avg_cos.getMean(),   trans.y,
                                                   0,                       0,         1);
                tf_it->tf.T = Tt;
                break;
            }
        }
    }
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

                //search in all the seg_ext(t-1) neighbours
                for(std::vector<NeighDataExt>::iterator s_o_neigh = s_o->second.begin(); s_o_neigh != s_o->second.end(); s_o_neigh++){
                    if((s_o_neigh->neigh->solved)||(!s_o_neigh->has_tf)){ continue; }//if extracted already continue

                    s_o->first->solved = true; s_o->first->getParrent()->solved = true; //flag seg_ext(t-1) and seg_init(t-1) from pair as extracted

                    s_o_neigh->neigh->solved = true; s_o_neigh->neigh->getParrent()->solved = true; //flag seg_ext(t) and seg_init(t) from pair as extracted

                    list_comm.push_back(CorrInput(s_o->first, s_o_neigh->neigh,0,0));//append the pair in the common list

                    //search in seg_ext(t) for the appended segments and append other possible links that have objects not yet in o_comm
                    for(std::vector<NeighDataExt>::iterator s_n = neigh_data_ne[s_o_neigh->neigh].begin(); s_n != neigh_data_ne[s_o_neigh->neigh].end(); s_n++){
                        if((s_n->neigh->solved)||(!s_n->has_tf)||(s_o->first == s_n->neigh)){ continue; }//if extracted already continue

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
