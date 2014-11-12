#include "data_processing/correlation.h"
#include "utils/math.h"
#include "utils/iterators.h"

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
            for(std::vector<NeighData>::iterator it_neigh = neigh_data[FRAME_NEW][it_corr->frame_new].begin(); it_neigh != neigh_data[FRAME_NEW][it_corr->frame_new].end(); it_neigh++){
                if( it_neigh->neigh == it_corr->frame_old ){
                    neigh_data[FRAME_NEW][it_corr->frame_new].erase(neigh_data[FRAME_NEW][it_corr->frame_new].begin() + index);
                    if(neigh_data[FRAME_NEW][it_corr->frame_new].size() == 0){
                        neigh_data[FRAME_NEW].erase(it_corr->frame_new);
                    }
                    break;
                }
                index++;
            }
            index = 0;
            if(neigh_data[FRAME_OLD].count(it_corr->frame_old) == 0){
                continue;
            }
            for(std::vector<NeighData>::iterator it_neigh = neigh_data[FRAME_OLD][it_corr->frame_old].begin(); it_neigh != neigh_data[FRAME_OLD][it_corr->frame_old].end(); it_neigh++){
                if( it_neigh->neigh == it_corr->frame_new ){
                    neigh_data[FRAME_OLD][it_corr->frame_old].erase(neigh_data[FRAME_OLD][it_corr->frame_old].begin() + index);
                    if(neigh_data[FRAME_OLD][it_corr->frame_old].size() == 0){
                        neigh_data[FRAME_OLD].erase(it_corr->frame_old);
                    }
                    break;
                }
                index++;
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::run(SensorData& sensor){
    if(sensor.status != OLD_FRAME){
        return;
    }
    corr_list.clear();
    neigh_data[0].clear();
    neigh_data[1].clear();

    calc_stitch_perc(sensor.frame_old->seg_ext,sensor.frame_new->seg_ext, FRAME_OLD);
    calc_stitch_perc(sensor.frame_new->seg_ext,sensor.frame_old->seg_ext, FRAME_NEW);

    merge_neigh_lists(FRAME_OLD);
    merge_neigh_lists(FRAME_NEW);

    resolve_weak_links(FRAME_OLD);
    resolve_weak_links(FRAME_NEW);

    set_flags(sensor.frame_old->seg_ext,sensor.frame_new->seg_ext);

    debug_cout_neigh_list(FRAME_OLD);
    debug_cout_neigh_list(FRAME_NEW);

    create_corr_queue();

    debug_cout_corr_queue(corr_list);

    run_conv(sensor);

//    std::cout<<"AFTER CONV !!! "<<std::endl;
    update_neigh_list();
//    debug_cout_neigh_list(FRAME_OLD);
//    debug_cout_neigh_list(FRAME_NEW);
//    create_corr_queue();
//    debug_cout_corr_queue(corr_list);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::run_conv(SensorData& sensor){
    if(sensor.status != OLD_FRAME){
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

void Correlation::plot_all_data(SensorData& sensor, cv::Scalar color_old, cv::Scalar color_new, KalmanSLDM k){
    if(sensor.status != OLD_FRAME){
        return;
    }   
    if(viz_corr_links){
        plot_data.plot_corr_links(corr_list,color_old,color_new);
    }
    if(viz_data_tf){
        plot_data.plot_segm_tf(sensor.frame_old->seg_ext, 0 , plot_data.blue);
        plot_data.plot_segm_tf(sensor.frame_new->seg_ext, 0 , plot_data.blue);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::create_corr_queue(){
//    std::cout<<"creating corr list"<<std::endl;

    corr_list.clear();

    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg = neigh_data[FRAME_OLD].begin(); it_seg != neigh_data[FRAME_OLD].end(); it_seg++){
        if( it_seg->second.size() == 0 ){
            insert_in_corr_queue( it_seg->first);
        }
        else{
            for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
                insert_in_corr_queue(it_seg->first,*it_neigh);
            }
        }
    }
    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg = neigh_data[FRAME_NEW].begin(); it_seg != neigh_data[FRAME_NEW].end(); it_seg++){
        if( it_seg->second.size() == 0 ){
            insert_in_corr_queue( it_seg->first);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::insert_in_corr_queue(const SegmentDataExtPtr seg_p_old, const NeighData& neigh_new){
    bool reverse= false;

    if(seg_p_old->len > neigh_new.neigh->len){ // setting direction based on the object with smaller length
        reverse = true;
    }

    CorrPairFlag new_pair_flag = corr_pair_flag(seg_p_old->corr_flag, neigh_new.neigh->corr_flag);
    double       new_pair_prob = std::fmax(neigh_new.prob_fwd, neigh_new.prob_rev);
    std::vector<CorrInput>::iterator it_corr;
    for(it_corr = corr_list.begin(); it_corr != corr_list.end(); it_corr++){
        CorrPairFlag corr_list_pair_flag = corr_pair_flag(it_corr->frame_old->corr_flag, it_corr->frame_new->corr_flag);
        if      (new_pair_flag < corr_list_pair_flag){//first sort priority on correlation type
            break;
        }
        else if((new_pair_flag == corr_list_pair_flag)&&( new_pair_prob > it_corr->stitch_perc + 0.05)){//2nd sort priority on snap percentage
            break;
        }
        else if((new_pair_flag == corr_list_pair_flag)&&( new_pair_prob < it_corr->stitch_perc + 0.05 )
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

void Correlation::set_flags(SegmentDataExtPtrVectorPtr &input_old,SegmentDataExtPtrVectorPtr &input_new){
    for(SegmentDataExtPtrVectorIter seg_it_old = input_old->begin(); seg_it_old != input_old->end(); seg_it_old++){
        if     ( neigh_data[FRAME_OLD][*seg_it_old].size() == 0 ){
            (*seg_it_old)->corr_flag = CORR_NOINNV;
        }
        else if( neigh_data[FRAME_OLD][*seg_it_old].size() == 1){
            (*seg_it_old)->corr_flag = CORR_121;
        }
        else if( neigh_data[FRAME_OLD][*seg_it_old].size() > 1){
            (*seg_it_old)->corr_flag = CORR_12MANY;
        }
    }
    for(SegmentDataExtPtrVectorIter seg_it_new = input_new->begin(); seg_it_new != input_new->end(); seg_it_new++){
        if     ( neigh_data[FRAME_NEW][*seg_it_new].size() == 0 ){
            (*seg_it_new)->corr_flag = CORR_NEWOBJ;
        }
        else if( neigh_data[FRAME_NEW][*seg_it_new].size() == 1 ){
            (*seg_it_new)->corr_flag = CORR_121;
        }
        else if( neigh_data[FRAME_NEW][*seg_it_new].size() > 1 ){
            (*seg_it_new)->corr_flag = CORR_MANY21;
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::resolve_weak_links(FrameStatus fr_status){
    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg = neigh_data[fr_status].begin(); it_seg != neigh_data[fr_status].end(); it_seg++){
        for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
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
    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg = neigh_data[fr_status].begin(); it_seg != neigh_data[fr_status].end(); it_seg++){
        for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin(); it_neigh != it_seg->second.end(); it_neigh++){
            std::vector<NeighData>::iterator it_neigh_2;
            for(it_neigh_2 = neigh_data[!fr_status][it_neigh->neigh].begin(); it_neigh_2 != neigh_data[!fr_status][it_neigh->neigh].end(); it_neigh_2++){
                if( it_neigh_2->neigh == it_seg->first ){
                    it_neigh->prob_rev   = it_neigh_2->prob_fwd;
                    it_neigh_2->prob_rev = it_neigh->prob_fwd;
                    break;
                }
            }
            if( it_neigh_2 == neigh_data[!fr_status][it_neigh->neigh].end() ){
                neigh_data[!fr_status][it_neigh->neigh].push_back(NeighData(it_seg->first,0,it_neigh->prob_fwd));
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

CorrPairFlag Correlation::corr_pair_flag( CorrFlag flag_old, CorrFlag flag_new ){
    if((flag_old == CORR_NOINNV)&&(flag_new == CORR_NOINNV)){
        return CORR_NO_CORR_OLD;
    }
    else if((flag_old == CORR_NEWOBJ)&&(flag_new == CORR_NEWOBJ)){
        return CORR_NO_CORR_NEW;
    }
    else if(( flag_old == CORR_121 )&&( flag_new == CORR_121 )){
        return CORR_SINGLE2;
    }
    else if(((( flag_old == CORR_121 )&&( flag_new == CORR_MANY21 )))||((( flag_old == CORR_12MANY )&&( flag_new == CORR_121 )))){
        return CORR_SINGLE1MULTI1;
    }
    else{
        return CORR_MULTI2;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::calc_stitch_perc(const SegmentDataExtPtrVectorPtr &input_ref, const SegmentDataExtPtrVectorPtr &input_spl, FrameStatus fr_status){
    neigh_data[fr_status].clear();
    IteratorIndexSet  iis_ref(input_ref);
    IteratorIndexSet2 iis_spl(input_spl);
    if( ( iis_spl.status() >= IIS2_VALID )&&( iis_spl.min().status() >= IIS_VALID )&&(iis_ref.status() >= IIS_VALID) ){
        do{
            double circle_rad = neigh_circle_rad;
            double ang_bounds[2];
            angular_bounds(*iis_ref.p(),circle_rad, ang_bounds);

            if( iis_spl.advance_in_ang_bounds(ang_bounds)){
                IteratorIndexSet iis_j;
                SegmentDataExtPtrVectorIter seg_min = iis_spl.min().input()->end();
                bool found_one = false;

                while( iis_spl.advance_divergent(iis_j,ang_bounds) ){
                    if( diff( *iis_j.p() , *iis_ref.p() ) <= circle_rad ){
                        circle_rad = diff( *iis_j.p() , *iis_ref.p() );//retine iteratorul minim intro var sa o inserezi jos vv
                        angular_bounds(*iis_ref.p(),circle_rad, ang_bounds);
                        seg_min = iis_j.seg();
                        found_one=true;

                    }
                }
                if(found_one){
                    std::vector<NeighData>::iterator it_neigh_data = neigh_data[fr_status][*iis_ref.seg()].begin();
                    while(it_neigh_data != neigh_data[fr_status][*iis_ref.seg()].end() ){
                        if( *seg_min == it_neigh_data->neigh ){
                            it_neigh_data->prob_fwd += 1.0 / (double)(*iis_ref.seg())->p.size() ;
                            break;
                        }
                        it_neigh_data++;
                    }
                    if( it_neigh_data == neigh_data[fr_status][*iis_ref.seg()].end() ){
                        neigh_data[fr_status][*iis_ref.seg()].push_back(NeighData(*seg_min, 1.0 / (double)(*iis_ref.seg())->p.size(), 0.0));
                    }
                }
            }
        }while(iis_ref.advance(ALL_SEGM,INC));
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

const char* CorrFlagNames[] =
  {
  stringify( CORR_121 ),
  stringify( CORR_12MANY ),
  stringify( CORR_MANY21 ),
  stringify( CORR_NOINNV ),
  stringify( CORR_NEWOBJ )
  };

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_segment(const SegmentDataExtPtr it_seg,bool old){
    std::cout<<"("<<std::setw(11)<<CorrFlagNames[it_seg->corr_flag]<<")(s:"<<std::setw(2)<<it_seg->id<<"o:"<<std::setw(2)<<it_seg->parrent->parrent->id<<")";
    if(old){std::cout<<"(t-1)";}
    else{std::cout<<"( t )";}
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_corr_queue(std::vector<CorrInput>& corr_queue){
    std::cout<<"---------------------------------------------------------------------------------------------------------------------------------------CORR QUEUE"<<std::endl;
    std::cout.precision(4);
    for(std::vector<CorrInput>::iterator it_corr = corr_queue.begin(); it_corr != corr_queue.end(); it_corr++){
        debug_cout_segment(it_corr->frame_old,true);
        if( it_corr->reverse ==true ){
            std::cout<<" <- ";
        }
        else{
            std::cout<<" -> ";
        }
        debug_cout_segment(it_corr->frame_new,false);
        std::cout<<"   "<<std::setw(5)<<it_corr->stitch_perc*100.0<<"%";

        std::cout<<"  l(t-1):"<<std::setw(6)<<it_corr->frame_old->len<<" l( t ):"<<std::setw(6)<<it_corr->frame_new->len;
        std::cout<<std::endl;
    }
    std::cout.precision(6);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Correlation::debug_cout_neigh_list(FrameStatus fr_status){
    if(fr_status == FRAME_OLD){
        std::cout<<"---------------------------------------------------------------------------------------------------------------------------------------t-1 FRAME NEIGHBOURS "<<std::endl;
    }
    else{
        std::cout<<"--------------------------------------------------------------------------------------------------------------------------------------- t  FRAME NEIGHBOURS "<<std::endl;
    }
    std::cout.precision(4);
    for(std::map <SegmentDataExtPtr, std::vector<NeighData> >::iterator it_seg = neigh_data[fr_status].begin();it_seg != neigh_data[fr_status].end();it_seg++){

        debug_cout_segment(it_seg->first,fr_status);
        std::cout<<" -> ";
        for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            debug_cout_segment(it_neigh->neigh,!fr_status);
            std::cout<<" | ";
        }
        std::cout<<std::endl<<"                                            ";

        for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"->"<<std::setw(5)<<std::fixed<<it_neigh->prob_fwd*100.0<<    "%                    ";
        }
        std::cout<<std::endl<<"                                            ";
        for(std::vector<NeighData>::iterator it_neigh = it_seg->second.begin();it_neigh != it_seg->second.end();it_neigh++){
            std::cout<<"<-"<<std::setw(5)<<std::fixed<<it_neigh->prob_rev*100.0<<"%                    ";
        }
        std::cout<<std::endl;
    }
    std::cout.precision(6);
}
