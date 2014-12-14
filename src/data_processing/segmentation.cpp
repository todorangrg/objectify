#include "data_processing/segmentation.h"
#include "utils/math.h"
#include "utils/iterators.h"
#include "utils/kalman.h"

extern const char* ConvFlagNames[];

//     LATER : comment, final cleanup, speed-up

Segmentation::Segmentation(RecfgParam &_param, SensorTf& _tf_sns, PlotData& _plot, KalmanSLDM &_k) :
    outl_circle_rad(_param.segm_outl_circle_rad),
    outl_sigma(_param.segm_outl_sigma),
    outl_prob_thres(_param.segm_outl_prob_thres),
    segm_discont_dist(_param.segm_discont_dist),
    sensor_range_max(_param.sensor_r_max),
    angle_inc(_param.cb_sensor_point_angl_inc),
    angle_max(_param.cb_sensor_point_angl_max),
    angle_min(_param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns),
    plot(_plot),
    plot_data_segm_init(_param.viz_data_segm_init),
    plot_data_segm_ext (_param.viz_data_segm_ext),
    k(_k){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::plot_data(InputData &input, KalmanSLDM k){
    if(plot_data_segm_init){
        if(k.seg_init){
            plot.plot_segm(k.seg_init    , plot.blue_dark);
        }
        if(input.seg_init){
            plot.plot_segm(input.seg_init, plot.red_dark );
        }
    }
    if(plot_data_segm_ext){
        if(k.seg_ext){
            plot.plot_segm(k.seg_ext    , plot.blue);
        }
        if(input.seg_ext){
            plot.plot_segm(input.seg_ext, plot.red);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::run(InputData &input, KalmanSLDM &k, bool advance){
    if(!k.seg_init){
        if(input.is_valid){
            assign_seg_init(input.sensor_filtered,input.seg_init);
            split_com_len(input.seg_init, false);
        }
        return;
    }
    tf_frm.init(k.rob_x_old(), k.rob_x_now());

    assign_seg_init(input.sensor_filtered,input.seg_init);
    split_com_len  (input.seg_init,      false);

    assign_seg_ext    (input.seg_init, input.seg_ext);
    calc_tf           (input.seg_ext , NEW2OLD);
    calc_occlusion    (input.seg_ext , IN_ALL );
    sample_const_angle(input.seg_ext );
    erase_img_outl    (input.seg_ext );
    split_com_len     (input.seg_ext,    false);
    calc_tf           (input.seg_ext , OLD2NEW);
    sample_const_angle(input.seg_ext );
    split_com_len     (input.seg_ext,    false);
    link_init_ext     (input.seg_ext );


    sort_seg_init     (k.seg_init);
    split_com_len     (k.seg_init,     true);

    assign_seg_ext    (k.seg_init, k.seg_ext);
    calc_tf           (k.seg_init,  OLD2NEW);
    split_for_occl    (k.seg_ext );
    calc_occlusion    (k.seg_ext ,  IN_SEG );
    calc_tf           (k.seg_ext ,  OLD2NEW);
    calc_occlusion    (k.seg_ext ,  IN_ALL );
    sample_const_angle(k.seg_ext );
    erase_img_outl    (k.seg_ext );
    split_com_len     (k.seg_ext ,    false);
    link_init_ext     (k.seg_ext );
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool ang_sort_func    (SegmentDataPtr i, SegmentDataPtr j) { return (i->p.front().angle    < j->p.front().angle ); }

void Segmentation::sort_seg_init(SegmentDataPtrVectorPtr &segments_init){
    std::sort(segments_init->begin(), segments_init->end(), ang_sort_func);
    int k = 0;
    for(SegmentDataPtrVectorIter it_in = segments_init->begin(); it_in != segments_init->end() ; it_in++){
        (*it_in)->id = k++;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool Segmentation::in_range(polar p){
    if( ( p.r < sensor_range_max ) && ( p.angle <= angle_max ) && ( p.angle >= angle_min ) ){
        return true;
    }
    return false;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::assign_seg_init(const PointDataVectorPtr &input , SegmentDataPtrVectorPtr &segments_init){
    segments_init->clear();
    if(!((input)&&(input->size() > 0))){//if invalid or empty return
        return;
    }
    segments_init->push_back(SegmentDataPtr(new SegmentData(0)));
    for( int i = 1 ; i < input->size() ; i++ ){
        segments_init->back()->p.push_back(input->at(i-1));
        if ( diff( input->at(i-1) , input->at(i) ) > segm_discont_dist ){//if split distance insert in new segment
            segments_init->push_back(SegmentDataPtr(new SegmentData(segments_init->back()->id+1)));
        }
    }
    segments_init->back()->p.push_back(input->back());
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::assign_seg_ext(const SegmentDataPtrVectorPtr &input, SegmentDataExtPtrVectorPtr &output){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    int id = 0;
    for(SegmentDataPtrVectorIter it_in = input->begin(); it_in != input->end() ; it_in++){
        temp->push_back(SegmentDataExtPtr(new SegmentDataExt(*it_in, id)));
        int p_added = 0;
        for(PointDataVectorIter it_p = (*it_in)->p.begin(); it_p != (*it_in)->p.end(); it_p++){
            if(in_range(*it_p)){
                temp->back()->p.push_back(*it_p);
                p_added++;
            }
        }
        if(p_added == 0){
            temp->pop_back();
        }
    }
    output = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

template <class SegData>
void Segmentation::calc_tf(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &input, TFmode tf_mode ){
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = input->begin(); ss != input->end(); ss++){
        for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
            polar tf_point;
            if     ( tf_mode == OLD2NEW ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.ro2rn( tf_sns.s2r( to_xy( *pp ) ) ) ) );
            }
            else if( tf_mode == NEW2OLD ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.rn2ro( tf_sns.s2r( to_xy( *pp ) ) ) ) );
            }
            normalizeAngle( tf_point.angle );
            pp->r     = tf_point.r;
            pp->angle = tf_point.angle;
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
// Removes also out of range points
// Uses iterators
void Segmentation::calc_occlusion(SegmentDataExtPtrVectorPtr &input, OcclType occ_type ){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    IteratorIndexSet<SegmentDataExt> iis(input);
    if(iis.status() >= IIS_VALID){
        polar angle_max( *iis.p() );
        bool exit = false;
        while(!in_range(*iis.p())){
            if(!iis.advance(ALL_SEGM, INC)){ exit = true; break; }
            if(iis.status() == IIS_SEG_END){ exit = true; break; }
        }
        if(!exit){ iis.push_bk(temp,*iis.p()); }
        while(iis.advance(ALL_SEGM, INC)){
            iis.advance(ALL_SEGM, DEC);//decrement
            if( angle_max.angle < iis.p()->angle ){
                angle_max = polar( *iis.p() );
            }
            SegmentDataExtPtrVectorIter seg1 = iis.seg();

            iis.advance(ALL_SEGM, INC);//increment

            if(( occ_type == IN_SEG ) && ( seg1 != iis.seg() )){
                angle_max = polar( *iis.p() );
                seg1 = iis.seg();
            }
            if(in_range(*iis.p())){
                if( iis.p()->angle < angle_max.angle ){
                    polar angle_min = polar( *iis.p() );
                    if( iis.p()->r < angle_max.r ){
                        while(( temp->size() > 0 ) || (( temp->size() > 0 ) && ( occ_type == IN_SEG ) && ( seg1 == --temp->end() ) ) ){
                            if( angle_min.angle >= temp->back()->p.back().angle ){
                                break;
                            }
                            iis.pop_bk(temp);
                        }
                        iis.push_bk(temp,angle_min );
                        angle_max = angle_min;
                    }
                }
                else{
                    iis.push_bk(temp,*iis.p() );
                }
            }
        }
    }
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
//check more but now should work good HERE PROBLEM SPLITTING WHEN SHOULD NOT!!!!!
void Segmentation::split_for_occl(SegmentDataExtPtrVectorPtr &input){
    input->reserve(2 * input->size());
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());

    int id=0;
    for(SegmentDataExtPtrVectorIter seg_inp_n = input->begin(); seg_inp_n != input->end(); seg_inp_n++){
        bool split_segment=false;
        int diff_pn = 1;
        for(SegmentDataExtPtrVectorIter seg_inp_p = seg_inp_n + 1; seg_inp_p != input->end(); seg_inp_p++){
            if((*seg_inp_n)->p.back().angle > (*seg_inp_p)->p.back().angle + 0.05){//hardcoded
                //split seg_inp_n and put it after seg_inp_p and break and continue
                PointDataVectorIter p = (*seg_inp_n)->p.begin();

                double mid_ang = ((*seg_inp_p)->p.back().angle + (*seg_inp_p)->p.front().angle) / 2.0;
                if((p != (*seg_inp_n)->p.end())&&( p->angle < normalizeAngle(mid_ang) )){
                    temp->push_back(SegmentDataExtPtr(new SegmentDataExt(seg_inp_n, id)));//copy till breaking point

                    while((p != (*seg_inp_n)->p.end())&&( p->angle < normalizeAngle(mid_ang) )){
                        temp->back()->p.push_back(PointData(*p));
                        p++;
                    }
                }
                //appending in input the remaining of the segment after seg_inp_p segment
                if(p != (*seg_inp_n)->p.end()){
                    PointDataVector points_to_append;
                    while(p != (*seg_inp_n)->p.end()){
                        points_to_append.push_back(PointData(*p));
                        p++;
                    }
                    SegmentDataExtPtrVectorIter inp_append = input->insert(seg_inp_p + 1, SegmentDataExtPtr(new SegmentDataExt(seg_inp_n)));
                    (*inp_append)->p = points_to_append;
                    seg_inp_p = --inp_append;
                    int diff_pnc = diff_pn;
                    while(diff_pnc > 0){ --inp_append; diff_pnc--; }
                    seg_inp_n = inp_append;
                }
                split_segment=true;
                break;
            }
            diff_pn++;
        }
        if(!split_segment){
            PointDataVectorIter p = (*seg_inp_n)->p.begin();
            if(p != (*seg_inp_n)->p.end()){
                temp->push_back(SegmentDataExtPtr(new SegmentDataExt(seg_inp_n, id)));//normal copy
                for(p = (*seg_inp_n)->p.begin(); p != (*seg_inp_n)->p.end(); p++){
                    temp->back()->p.push_back(PointData(*p));
                }
            }
        }
        id++;
    }
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
// Uses iterators
void Segmentation::sample_const_angle(SegmentDataExtPtrVectorPtr &input){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(2*input->size());//heuristic preallocation

    std::vector<bool> quant_pos((2 * M_PI) / angle_inc,false);

    IteratorIndexSet<SegmentDataExt> iis0(input);
    IteratorIndexSet<SegmentDataExt> iis1(input);
    if(iis1.status() < IIS_VALID){
        return;
    }
    iis1.advance(ALL_SEGM, INC);
    if(iis1.status() >= IIS_VALID){
        do{
            double disc_angle0 = ( iis0.p()->angle + M_PI);
            double disc_angle1 = ( iis1.p()->angle + M_PI );
            double sample_pos_neg = round(disc_angle0 / angle_inc ) ;
            double sample_pos_pos = round(disc_angle1 / angle_inc ) ;
            disc_angle0 = iis0.p()->angle;
            disc_angle1 = iis1.p()->angle;
            normalizeAngle(disc_angle0);

            polar sample( 0 , disc_angle0);
            if( iis0.seg() == iis1.seg() ){
                if( diff( *iis0.p() , *iis1.p() ) < segm_discont_dist ){

                    do{
                        if( quant_pos[sample_pos_neg] == false ){
                            quant_pos[sample_pos_neg] = true;

                            Line l_input = get_line_param( to_xy( *iis0.p() ) , to_xy( *iis1.p() ) );
                            Line l_angle = get_line_param( xy( 0 , 0 ) , to_xy( polar( sensor_range_max , sample.angle ) ) );
                            sample.r = diff( polar( 0 , 0 ) , to_polar( get_line_inters( l_input , l_angle ) ) );///NOT totally efficient

                            iis0.push_bk(temp, PointData(sample));
                        }
                        sample_pos_neg++;
                        sample = polar( 0 , sample.angle + angle_inc  );
                    }while( sample_pos_neg < sample_pos_pos/* - quant_err */);
                }
                else{
                    int x=0;
                }
            }
//            else if ( quant_pos[sample_pos_neg] == false ){
//                quant_pos[sample_pos_neg] = true;
//                sample.r = iis0.p()->r;
//                iis0.push_bk(temp, PointData( sample));
//            }
//            if ( ( iis1.p() == --input->back()->p.end() ) && ( quant_pos[sample_pos_pos] == false ) ){
//                quant_pos[sample_pos_pos] = true;
//                sample.r = iis1.p()->r;
//                iis0.push_bk(temp, PointData( polar(sample.r,sample.angle)));
//            }
        }while(iis1.advance(ALL_SEGM, INC) && iis0.advance(ALL_SEGM, INC));
    }
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
// Uses iterators
void Segmentation::erase_img_outl(SegmentDataExtPtrVectorPtr &input){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    std::vector<bool> temp_valid;

    IteratorIndexSet<SegmentDataExt> iis(input);
    if(iis.status() >= IIS_VALID){
        do{
            iis.push_bk(temp,*iis.p());
            temp_valid.push_back(true);
            check_neigh_p( input, temp , temp_valid ,iis);
        }while( iis.advance(ALL_SEGM, INC));


    }
    iis=IteratorIndexSet<SegmentDataExt>(temp,REV);
    if(iis.status() >= IIS_VALID){
        std::vector<bool>::reverse_iterator it = temp_valid.rbegin() ;
        do{
            if( !*it ){
                iis.erase();
            }
            it++;
        }while( iis.advance(ALL_SEGM, DEC));
    }
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
// Uses iterators
void Segmentation::check_neigh_p(const SegmentDataExtPtrVectorPtr &input, SegmentDataExtPtrVectorPtr &temp, std::vector<bool> &temp_valid, IteratorIndexSet<SegmentDataExt> iis){
    std::vector<IteratorIndexSet<SegmentDataExt> > neigh_seg;
    double p_prob = 0;
    polar p_i( *iis.p() );
    double search_angle;
    if( fabs( outl_circle_rad / p_i.r ) > 1.0  ){
        search_angle = 2 * M_PI;
    }
    else{
        search_angle = asin( outl_circle_rad / p_i.r );
    }
    for( int dir = -1 ; dir < 2 ; dir+=2 ){
        IteratorIndexSet<SegmentDataExt> iis_j = iis;

        while( iis_j.advance( ALL_SEGM, IISmode(std::max(dir,0)))){
            if(!( ( iis_j.p()->angle >= p_i.angle - search_angle ) && ( iis_j.p()->angle <= p_i.angle + search_angle ) &&( ( temp_valid[iis_j.i()] == true ) || ( iis_j.i() >= temp_valid.size() ) ))){
                break;
            }
            polar p_j = polar( *iis_j.p() );
            neigh_seg.push_back(iis_j);
            if( diff( p_j , p_i ) <= outl_circle_rad ){
                p_prob  += std::exp( - ( sqr( diff( p_j , p_i ) ) ) / ( outl_sigma * sqr( outl_circle_rad ) ) );
                if( p_prob >= outl_prob_thres ){
                    break;
                }
            }
        }
    }
    if( p_prob < outl_prob_thres ){
        temp_valid[ iis.i() ] = false;
        int temp_size=0;
        for(SegmentDataExtPtrVectorIter temp_it=temp->begin();temp_it!=temp->end();temp_it++){
            temp_size+=(*temp_it)->p.size();
        }
        for(std::vector<IteratorIndexSet<SegmentDataExt> >::iterator neigh_seg_it=neigh_seg.begin();neigh_seg_it != neigh_seg.end();neigh_seg_it++){
            if(neigh_seg_it->i() < temp_size){
                check_neigh_p( input , temp , temp_valid ,*neigh_seg_it);
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
//TODO not really efficient
void Segmentation::link_init_ext(SegmentDataExtPtrVectorPtr &ext){

    for(SegmentDataExtPtrVectorIter inp = ext->begin(); inp != ext->end(); inp++){

        PointData p_ext[2] = {(*inp)->p.front(), (*inp)->p.back()};

        PointDataVectorIter bound_low  = (*inp)->getParrent()->p.begin();
        PointDataVectorIter bound_high = (*inp)->getParrent()->p.end();
        double d_min[2] = {10000, 10000};
        int p=0;
        int p_min[2] = {0,0};
        double d_in = 0;
        double d_neg = 0, d_pos = 0;
        for(PointDataVectorIter it_in = bound_low;it_in != bound_high;it_in++){
            if(it_in != bound_low){
                PointDataVectorIter it_min = it_in; it_min--;
                d_in += diff( *it_in, *it_min );
            }
            if(diff(*it_in, p_ext[0]) < d_min[0]){
                d_min[0] = diff(*it_in, p_ext[0]);
                p_min[0] = p;
                d_neg = d_in;
            }
            if(diff(*it_in, p_ext[1]) < d_min[1]){
                d_min[1] = diff(*it_in, p_ext[1]);
                p_min[1] = p;
                d_pos = d_in;
            }
            p++;
        }
        d_pos = d_in - d_pos;
        for(PointDataVectorIter it_in = bound_low + p_min[0];(it_in != bound_low + p_min[1] + 1);it_in++){
            it_in->child = (*inp);
        }

        (*inp)->len_init_neg = d_neg;
        (*inp)->len_init_pos = d_pos;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
// Uses iterators
template <class SegData>
void Segmentation::split_com_len(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, bool old_init){
    double sample_dist = 0.05;
    boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > temp( new std::vector<boost::shared_ptr<SegData> > );
    temp->reserve(_input->size());

    IteratorIndexSet2<SegData> iis2(_input);
    iis2.maj().advance(ALL_SEGM,INC);

    if(iis2.status() >=IIS2_VALID){
        bool split_segment=false;
        double len = 0;
        double rest=0;
        double p_nr = 0;
        xy com;
        xy p_now,p1_last,p2_last;
        do{
            iis2.min().push_bk(temp, *iis2.min().p(), split_segment);//add the min value in temp, splitting segment or not
            double d_min_maj = 0;
            if( iis2.min().seg() == iis2.maj().seg() ){//if min and maj on same segment
                split_segment = false;
                d_min_maj = diff( *iis2.min().p() , *iis2.maj().p() );
                if(  d_min_maj > segm_discont_dist ){
                    split_segment=true;
                    if(len > min_seg_dist){
                        if(p_nr == 0){ com = to_xy(temp->back()->p.back()); p_nr = 1; }
                        temp->back()->setLen(len);
                        temp->back()->setCom(xy(com.x / p_nr, com.y / p_nr));
                    }
                    else{ temp->pop_back(); }//here discards very short segments
                    len = 0; rest = 0; p_nr = 0; com = xy(0,0);
                }
                else{
                    len +=d_min_maj;
                    p1_last=to_xy(*iis2.min().p());
                    p2_last=to_xy(*iis2.maj().p());
                    while(rest+diff(to_polar(p1_last),*iis2.maj().p()) > sample_dist){

                        double k = (sample_dist-rest)/(diff(to_polar(p1_last),*iis2.maj().p()));
                        p_now = xy(p1_last.x+(p2_last.x-p1_last.x)*k,p1_last.y+(p2_last.y-p1_last.y)*k);

                        p1_last=p_now;

                        com += p_now;//com computation
                        p_nr++;

                        rest=0;
                    }
                    rest+=diff(to_polar(p1_last),*iis2.maj().p());
                }
            }
            else{
                split_segment=true;
                if(len > min_seg_dist){
                    if(p_nr == 0){ com = to_xy(temp->back()->p.back()); p_nr = 1;}
                    temp->back()->setLen(len);
                    temp->back()->setCom(xy(com.x / p_nr, com.y / p_nr));
                }
                else{ temp->pop_back(); }//here discards very short segments
                len = 0; rest = 0; p_nr = 0; com = xy(0,0);
            }
        }while(iis2.advance(ALL_SEGM,INC));
    }
    if(temp->size() == 1){
        if(temp->back()->p.size() == 1){//for one point
            temp->back()->setLen(0);
            temp->back()->setCom(xy(to_xy(temp->back()->p.back())));
            temp->pop_back();//here discards one point segments
        }
    }
    if(old_init){
        for(typename std::vector<boost::shared_ptr<SegData> >::iterator inp = temp->begin(); inp != temp->end(); inp++){
            (*inp)->setCom(tf_sns.r2s( tf_frm.ro2rn( tf_sns.s2r( (*inp)->getCom() ) ) ));
        }
    }
    _input=temp;
}
