#include "data_processing/correlation.h"
#include "utils/math.h"
#include "utils/iterators.h"
#include "utils/iterators_ss.h"
#include "utils/iterators_ss_tf.h"

Correlation::Correlation(RecfgParam &_param,PlotData& _plot_data,PlotConv& _plot_conv):
    Convolution(_param),
    queue_d_thres(_param.corr_queue_d_thres),
    neigh_circle_rad(_param.corr_neigh_circle_rad),
    plot_data(_plot_data),
    plot_conv(_plot_conv),
    viz_convol_all(_param.viz_convol_all),
    viz_convol_step_no(_param.viz_correl_queue_no),
    viz_corr_links(_param.viz_data_corr_links),
    viz_data_tf(_param.viz_data_tf){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::update_neigh_list(){// TODO INEFFICIENT FASTLY WRITTEN
    for(std::vector<CorrInput>::iterator it_corr = corr_list.begin() ; it_corr != corr_list.end(); it_corr++){
        if(it_corr->frame_old->conv->tf->size() == 0){
            int index = 0;
            for(std::vector<NeighDataExt>::iterator it_neigh = neigh_data_ext[FRAME_NEW][it_corr->frame_new].begin(); it_neigh != neigh_data_ext[FRAME_NEW][it_corr->frame_new].end(); it_neigh++){
                if( it_neigh->neigh == it_corr->frame_old ){
                    neigh_data_ext[FRAME_NEW][it_corr->frame_new].erase(neigh_data_ext[FRAME_NEW][it_corr->frame_new].begin() + index);
                    if(neigh_data_ext[FRAME_NEW][it_corr->frame_new].size() == 0){
                        neigh_data_ext[FRAME_NEW].erase(it_corr->frame_new);
                    }
                    break;
                }
                index++;
            }
            index = 0;
            if(neigh_data_ext[FRAME_OLD].count(it_corr->frame_old) == 0){
                continue;
            }
            for(std::vector<NeighDataExt>::iterator it_neigh = neigh_data_ext[FRAME_OLD][it_corr->frame_old].begin(); it_neigh != neigh_data_ext[FRAME_OLD][it_corr->frame_old].end(); it_neigh++){
                if( it_neigh->neigh == it_corr->frame_new ){
                    neigh_data_ext[FRAME_OLD][it_corr->frame_old].erase(neigh_data_ext[FRAME_OLD][it_corr->frame_old].begin() + index);
                    if(neigh_data_ext[FRAME_OLD][it_corr->frame_old].size() == 0){
                        neigh_data_ext[FRAME_OLD].erase(it_corr->frame_old);
                    }
                    break;
                }
                index++;
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::run(InputData &input, KalmanSLDM k, bool new_frame){
    if(!input.is_valid){
        return;
    }
    corr_list.clear();
    neigh_data_init[0].clear();
    neigh_data_init[1].clear();
    neigh_data_ext [0].clear();
    neigh_data_ext [1].clear();




    //calc_stitch_perc_ext(input.seg_ext,k.seg_ext, FRAME_NEW);
    calc_stitch_perc(input.seg_init,k.seg_init, FRAME_NEW);
    if(k.seg_init){
        calc_stitch_perc(k.seg_init,input.seg_init, FRAME_OLD);
        merge_neigh_lists(FRAME_OLD);
        merge_neigh_lists(FRAME_NEW);
        resolve_weak_links(FRAME_OLD);
    }
    resolve_weak_links(FRAME_NEW);

    if(k.seg_init){
        set_flags(k.seg_ext, FRAME_OLD);
    }
    set_flags(input.seg_ext, FRAME_NEW);


    if(new_frame){
        debug_cout_neigh_list(FRAME_OLD);
        debug_cout_neigh_list(FRAME_NEW);
    }

    create_corr_queue();

    if(new_frame){
        debug_cout_corr_queue(corr_list);
    }

    run_conv(input, k);


    update_neigh_list();
    if(new_frame){
        std::cout<<"AFTER CONV !!! "<<std::endl;
        debug_cout_neigh_list(FRAME_OLD);
        debug_cout_neigh_list(FRAME_NEW);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::run_conv(InputData &input, KalmanSLDM k){
    if(!input.is_valid){
        return;
    }

    for(std::vector<CorrInput>::iterator it = corr_list.begin(); it!= corr_list.end();it++){/// dummy reset on deletable p_const_dist
        it->frame_old->conv.reset();
        it->frame_new->conv.reset();
    }

    int iii=0;
    for(std::vector<CorrInput>::iterator it = corr_list.begin(); it!= corr_list.end();it++){
        if(create_normal_database(*it)){//!!
            convolute();//!!
            //SEE WHAT YOU DO IF NOTHING IS SMALLER THAN SCORE_THRES
            if( viz_convol_step_no == iii ){

                if(viz_convol_all){
                    plot_conv.plot_conv_info(conv_distr,plot_conv.blue,plot_conv.red,plot_conv.cyan);
                    plot_conv.plot_conv_points(conv_distr,conv_data);
                }
                else{
                    plot_conv.plot_conv_info(conv_accepted,plot_conv.blue,plot_conv.red,plot_conv.cyan);
                    plot_conv.plot_conv_points(conv_accepted,conv_data);
                }

            }
            fade_out_snapped_p();
            iii++;
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::plot_all_data(InputData &input, KalmanSLDM k, cv::Scalar color_old, cv::Scalar color_new){
    if(!input.is_valid){
        return;
    }
    if(viz_corr_links){
        plot_data.plot_corr_links(corr_list,color_old,color_new);
    }
    if(viz_data_tf){
        plot_data.plot_segm_tf(k.seg_ext, 0 , plot_data.blue);
        plot_data.plot_segm_tf(input.seg_ext, 0 , plot_data.blue);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::create_corr_queue(){
    //std::cout<<"creating corr list"<<std::endl;

    corr_list.clear();

    for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator it_seg = neigh_data_ext[FRAME_OLD].begin(); it_seg != neigh_data_ext[FRAME_OLD].end(); it_seg++){
        if( it_seg->second.size() != 0 ){
            for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
                insert_in_corr_queue(it_seg->first,*it_neigh);
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::insert_in_corr_queue(const SegmentDataExtPtr seg_p_old, const NeighDataExt& neigh_new){
    bool reverse= false;

    if(seg_p_old->len > neigh_new.neigh->len){ // setting direction based on the object with smaller length
        reverse = true;
    }

    int new_flag_weight = seg_p_old->corr_flag + neigh_new.neigh->corr_flag;
    double       new_pair_prob = std::fmax(neigh_new.prob_fwd, neigh_new.prob_rev);
    std::vector<CorrInput>::iterator it_corr;
    for(it_corr = corr_list.begin(); it_corr != corr_list.end(); it_corr++){
        int it_flag_weight = it_corr->frame_old->corr_flag + it_corr->frame_new->corr_flag;
        if      (new_flag_weight < it_flag_weight){//first sort priority on correlation type
            break;
        }
        else if((new_flag_weight == it_flag_weight)&&( new_pair_prob > it_corr->stitch_perc + 0.05)){//2nd sort priority on snap percentage
            break;
        }
        else if((new_flag_weight == it_flag_weight)&&( new_pair_prob < it_corr->stitch_perc + 0.05 )
                                                   &&( new_pair_prob > it_corr->stitch_perc - 0.05 )){//3rd sort priority on ref segment length
            double it_corr_len_ref;
            if( it_corr->reverse ){ it_corr_len_ref = it_corr->frame_new->len; }
            else                  { it_corr_len_ref = it_corr->frame_old->len; }
            double it_new_len_ref;
            if(reverse){ it_new_len_ref = neigh_new.neigh->len;}
            else       { it_new_len_ref = seg_p_old->len;}

            if( it_new_len_ref > it_corr_len_ref ){
                break;
            }
        }

    }
    it_corr=corr_list.insert(it_corr,CorrInput(seg_p_old, neigh_new.neigh, new_pair_prob, reverse));
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::insert_in_corr_queue(const SegmentDataExtPtr seg_p_old){
    corr_list.insert(corr_list.begin(),CorrInput(seg_p_old, seg_p_old, 0, 0));
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::set_flags(SegmentDataExtPtrVectorPtr &input, FrameStatus fr_stat){
    for(SegmentDataExtPtrVectorIter seg_it = input->begin(); seg_it != input->end(); seg_it++){
        if  ( neigh_data_ext[fr_stat][*seg_it].size() == 1){
            (*seg_it)->corr_flag = CORR_121;
        }
        else if( neigh_data_ext[fr_stat][*seg_it].size() > 1){
            (*seg_it)->corr_flag = CORR_12MANY;
        }
    }
}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::resolve_weak_links(FrameStatus fr_status){
    for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator it_seg = neigh_data_ext[fr_status].begin(); it_seg != neigh_data_ext[fr_status].end(); it_seg++){
        for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
            if( fmax(it_neigh->prob_fwd,it_neigh->prob_rev) < queue_d_thres ){
                it_neigh = it_seg->second.erase(it_neigh);
                it_neigh--;
            }
        }
    }
    for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator it_seg = neigh_data_init[fr_status].begin(); it_seg != neigh_data_init[fr_status].end(); it_seg++){
        for(std::vector<NeighDataInit>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
            if( fmax(it_neigh->prob_fwd,it_neigh->prob_rev) < queue_d_thres ){
                it_neigh = it_seg->second.erase(it_neigh);
                it_neigh--;
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::merge_neigh_lists(FrameStatus fr_status){
//    std::cout<<"merging lists"<<std::endl;
    for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator it_seg = neigh_data_ext[fr_status].begin(); it_seg != neigh_data_ext[fr_status].end(); it_seg++){
        for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
            std::vector<NeighDataExt>::iterator it_neigh_2;
            for(it_neigh_2 = neigh_data_ext[!fr_status][it_neigh->neigh].begin(); it_neigh_2 != neigh_data_ext[!fr_status][it_neigh->neigh].end(); it_neigh_2++){
                if( it_neigh_2->neigh == it_seg->first ){
                    it_neigh->prob_rev   = it_neigh_2->prob_fwd;
                    it_neigh_2->prob_rev = it_neigh->prob_fwd;
                    break;
                }
            }
            if( it_neigh_2 == neigh_data_ext[!fr_status][it_neigh->neigh].end() ){
                neigh_data_ext[!fr_status][it_neigh->neigh].push_back(NeighDataExt(it_seg->first,0,it_neigh->prob_fwd));
            }
        }
    }
    for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator it_seg = neigh_data_init[fr_status].begin(); it_seg != neigh_data_init[fr_status].end(); it_seg++){
        for(std::vector<NeighDataInit>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
            std::vector<NeighDataInit>::iterator it_neigh_2;
            for(it_neigh_2 = neigh_data_init[!fr_status][it_neigh->neigh].begin(); it_neigh_2 != neigh_data_init[!fr_status][it_neigh->neigh].end(); it_neigh_2++){
                if( it_neigh_2->neigh == it_seg->first ){
                    it_neigh->prob_rev   = it_neigh_2->prob_fwd;
                    it_neigh_2->prob_rev = it_neigh->prob_fwd;
                    break;
                }
            }
            if( it_neigh_2 == neigh_data_init[!fr_status][it_neigh->neigh].end() ){
                neigh_data_init[!fr_status][it_neigh->neigh].push_back(NeighDataInit(it_seg->first,0,it_neigh->prob_fwd));
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///


bool ang_sort_func    (PointDataCpy i, PointDataCpy j) { return (i.angle  < j.angle ); }
// TODO TODO TODO DEBUG IT!!!
void Correlation::calc_stitch_perc(const SegmentDataPtrVectorPtr &input_ref, const SegmentDataPtrVectorPtr &input_spl, FrameStatus fr_status){
    neigh_data_init[fr_status].clear();
    neigh_data_ext [fr_status].clear();
    for(SegmentDataPtrVectorIter seg_init = input_ref->begin(); seg_init != input_ref->end(); seg_init++){
        neigh_data_init[fr_status][*seg_init];
    }


    ///////////////////////////////////////////////////

    if((!input_ref)||(!input_spl)){
        return;
    }
    std::vector<PointDataCpy > p_array[2];
    for(SegmentDataPtrVectorIter seg = input_ref->begin(); seg != input_ref->end(); seg++){
//        if(fr_status == FRAME_OLD){
            for(PointDataVectorIter p = (*seg)->p_tf.begin(); p != (*seg)->p_tf.end(); p++){
                p_array[CONV_REF].push_back(PointDataCpy(*p, *seg));
            }
//        }
//        else{
//            for(PointDataVectorIter p = (*seg)->p.begin(); p != (*seg)->p.end(); p++){
//                p_array[CONV_REF].push_back(PointDataCpy(*p, *seg));
//            }
//        }
    }
    std::sort(p_array[CONV_REF].begin(), p_array[CONV_REF].end(), ang_sort_func);

    for(SegmentDataPtrVectorIter seg = input_spl->begin(); seg != input_spl->end(); seg++){
//        if(fr_status == FRAME_OLD){
//            for(PointDataVectorIter p = (*seg)->p.begin(); p != (*seg)->p.end(); p++){
//                p_array[CONV_SPL].push_back(PointDataCpy(*p, *seg));
//            }
//        }
//        else{
            for(PointDataVectorIter p = (*seg)->p_tf.begin(); p != (*seg)->p_tf.end(); p++){
                p_array[CONV_SPL].push_back(PointDataCpy(*p, *seg));
            }
//        }
    }
    std::sort(p_array[CONV_SPL].begin(), p_array[CONV_SPL].end(), ang_sort_func);

    ///if something return
//    if((p_array[CONV_SPL].size() == 0)||(p_array[CONV_REF].size() == 0)){
//        return;
//    }


    const int POS = 1;
    const int NEG = 0;

    int i_b[2] = {0, 0};
    for(std::vector<PointDataCpy>::iterator p_ref = p_array[CONV_REF].begin(); p_ref != p_array[CONV_REF].end(); p_ref++){
        double circle_rad = neigh_circle_rad;
        double ang_bounds[2];
        angular_bounds(*p_ref,circle_rad, ang_bounds);


        if(i_b[NEG] < 0 ){ i_b[NEG] = 0; }
        if(i_b[NEG] > p_array[CONV_SPL].size() - 1){ i_b[NEG] = p_array[CONV_SPL].size() - 1; }
        if(i_b[POS] < 0 ){ i_b[POS] = 0; }
        if(i_b[POS] > p_array[CONV_SPL].size() - 1){ i_b[POS] = p_array[CONV_SPL].size() - 1; }
        if((fabs(p_array[CONV_SPL][i_b[POS]].angle - ang_bounds[1]) > fabs(p_array[CONV_SPL][i_b[NEG]].angle - ang_bounds[0]))){
            while(p_array[CONV_SPL][i_b[POS]].angle > ang_bounds[1]){i_b[POS]--;}
            i_b[NEG] = i_b[POS];
        }
        else{
            while(p_array[CONV_SPL][i_b[NEG]].angle < ang_bounds[0]){i_b[NEG]++;}
            i_b[POS] = i_b[NEG];
        }
        bool found_one = false;
        PointData p_min;
        SegmentDataPtr seg_min;

        int out_of_bounds[2] = {0,0};
        while(1){
            for(int i = -1; i <= 1; i = i + 2){
                int i_now = i_b[std::max(0,i)] + i;

                if((i_now > p_array[CONV_SPL].size() - 1)||(i_now < 0)){
                    out_of_bounds[std::max(0,i)] = 1;
                    continue;
                }
                else if(((i == -1)&&(p_array[CONV_SPL][i_now].angle < ang_bounds[NEG]))||
                        ((i ==  1)&&(p_array[CONV_SPL][i_now].angle > ang_bounds[POS]))){
                    out_of_bounds[std::max(0,i)] = 1;
                    continue;
                }
                else{
                    out_of_bounds[std::max(0,i)] = 0;
                    i_b[std::max(0,i)] += i;
                }

                double dx = diff( polar(p_array[CONV_SPL][i_now].r, p_array[CONV_SPL][i_now].angle)  , polar(p_ref->r, p_ref->angle) );
                if(dx <= circle_rad ){
                    circle_rad = dx;//retine iteratorul minim intro var sa o inserezi jos vv
                    angular_bounds(*p_ref,circle_rad, ang_bounds);
                    p_min = p_array[CONV_SPL][i_now].p_parrent;
                    seg_min = p_array[CONV_SPL][i_now].s_parrent;
                    found_one=true;

                }

            }
            if((out_of_bounds[NEG] == 1)&&(out_of_bounds[POS] == 1)){
                break;
            }
        }
        if(found_one){
            p_ref->p_parrent.neigh = seg_min;
            std::vector<NeighDataInit>::iterator it_neigh_data = neigh_data_init[fr_status][p_ref->s_parrent].begin();
            while(it_neigh_data != neigh_data_init[fr_status][p_ref->s_parrent].end() ){
                if( seg_min == it_neigh_data->neigh ){
                    it_neigh_data->prob_fwd += 1.0 / (double)(p_ref->s_parrent->p.size()) ;
                    break;
                }
                it_neigh_data++;
            }
            if( it_neigh_data == neigh_data_init[fr_status][p_ref->s_parrent].end() ){
                neigh_data_init[fr_status][p_ref->s_parrent].push_back(NeighDataInit(seg_min, 1.0 / (double)p_ref->s_parrent->p.size(), 0.0));
            }

            if(p_ref->p_parrent.child.lock()){
                std::vector<NeighDataExt>::iterator it_neigh_data_ext = neigh_data_ext[fr_status][p_ref->p_parrent.child.lock()].begin();
                while(it_neigh_data_ext != neigh_data_ext[fr_status][p_ref->p_parrent.child.lock()].end() ){
                    if( p_min.child.lock() == it_neigh_data_ext->neigh ){
                        it_neigh_data_ext->prob_fwd += 1.0 / (double)p_ref->p_parrent.child.lock()->p.size() ;
                        break;
                    }
                    it_neigh_data_ext++;
                }
                if( it_neigh_data_ext == neigh_data_ext[fr_status][p_ref->p_parrent.child.lock()].end() ){
                    if(p_min.child.lock()){
                        neigh_data_ext[fr_status][p_ref->p_parrent.child.lock()].push_back(NeighDataExt(p_min.child.lock(), 1.0 / (double)p_ref->p_parrent.child.lock()->p.size(), 0.0));
                    }
                }
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

const char* CorrFlagNames[] =
  {
  stringify( CORR_121 ),
  stringify( CORR_12MANY )
  };

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_segment(SegmentDataBase& it_seg,bool old){
    std::cout<<"("<<std::setw(11)<<CorrFlagNames[it_seg.corr_flag]<<")(s:"<<std::setw(2)<<it_seg.id<<"o:"<<std::setw(2);
    if(it_seg.getObj()){
        std::cout<<it_seg.getObj()->id<<")";
    }
    if(old){std::cout<<"(t-1)";}
    else{std::cout<<"( t )";}
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_corr_queue(std::vector<CorrInput>& corr_queue){
    std::cout<<"---------------------------------------------------------------------------------------------------------------------------------------CORR QUEUE"<<std::endl;
    std::cout.precision(4);
    for(std::vector<CorrInput>::iterator it_corr = corr_queue.begin(); it_corr != corr_queue.end(); it_corr++){
        debug_cout_segment(*it_corr->frame_old,true);
        if( it_corr->reverse ==true ){
            std::cout<<" <- ";
        }
        else{
            std::cout<<" -> ";
        }
        debug_cout_segment(*it_corr->frame_new,false);
        std::cout<<"   "<<std::setw(5)<<it_corr->stitch_perc*100.0<<"%";

        std::cout<<"  l(t-1):"<<std::setw(6)<<it_corr->frame_old->len<<" l( t ):"<<std::setw(6)<<it_corr->frame_new->len;
        std::cout<<std::endl;
    }
    std::cout.precision(6);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_neigh_list(FrameStatus fr_status){
    if(fr_status == FRAME_OLD){
        std::cout<<"---------------------------------------------------------------------------------------------------------------------------------------t-1 FRAME NEIGHBOURS INIT "<<std::endl;
    }
    else{
        std::cout<<"--------------------------------------------------------------------------------------------------------------------------------------- t  FRAME NEIGHBOURS INIT"<<std::endl;
    }
    std::cout.precision(4);
    for(std::map <SegmentDataPtr, std::vector<NeighDataInit> >::iterator it_seg = neigh_data_init[fr_status].begin();it_seg != neigh_data_init[fr_status].end();it_seg++){

        debug_cout_segment(*it_seg->first,fr_status);
        std::cout<<" -> ";
        for(std::vector<NeighDataInit>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            debug_cout_segment(*it_neigh->neigh,!fr_status);
            std::cout<<" | ";
        }
        std::cout<<std::endl<<"                                            ";

        for(std::vector<NeighDataInit>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"->"<<std::setw(5)<<std::fixed<<it_neigh->prob_fwd*100.0<<    "%                    ";
        }
        std::cout<<std::endl<<"                                            ";
        for(std::vector<NeighDataInit>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"<-"<<std::setw(5)<<std::fixed<<it_neigh->prob_rev*100.0<<"%                    ";
        }
        std::cout<<std::endl;
    }

    if(fr_status == FRAME_OLD){
        std::cout<<"---------------------------------------------------------------------------------------------------------------------------------------t-1 FRAME NEIGHBOURS EXTENDED"<<std::endl;
    }
    else{
        std::cout<<"--------------------------------------------------------------------------------------------------------------------------------------- t  FRAME NEIGHBOURS EXTENDED"<<std::endl;
    }
    for(std::map <SegmentDataExtPtr, std::vector<NeighDataExt> >::iterator it_seg = neigh_data_ext[fr_status].begin();it_seg != neigh_data_ext[fr_status].end();it_seg++){

        debug_cout_segment(*it_seg->first,fr_status);
        std::cout<<" -> ";
        for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            debug_cout_segment(*it_neigh->neigh,!fr_status);
            std::cout<<" | ";
        }
        std::cout<<std::endl<<"                                            ";

        for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"->"<<std::setw(5)<<std::fixed<<it_neigh->prob_fwd*100.0<<    "%                    ";
        }
        std::cout<<std::endl<<"                                            ";
        for(std::vector<NeighDataExt>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"<-"<<std::setw(5)<<std::fixed<<it_neigh->prob_rev*100.0<<"%                    ";
        }
        std::cout<<std::endl;
    }
    std::cout.precision(6);
}
