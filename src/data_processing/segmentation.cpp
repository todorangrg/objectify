#include "data_processing/segmentation.h"
#include "utils/math.h"
#include "utils/iterators.h"
#include "utils/kalman.h"
#include "utils/iterators_ss.h"

extern const char* ConvFlagNames[];


//TODO sort the recieved old-frames to be still angular sorted

enum{
    SEG_P,
    SEG_P_TF
};


Segmentation::Segmentation(RecfgParam &_param, SensorTf& _tf_sns, PlotData& _plot, KalmanSLDM &_k) :
    outl_circle_rad(_param.segm_outl_circle_rad),
    outl_sigma(_param.segm_outl_sigma),
    outl_prob_thres(_param.segm_outl_prob_thres),
    segm_discont_dist(_param.segm_discont_dist),
    sensor_range_max(_param.sensor_range_max),
    angle_inc(_param.cb_sensor_point_angl_inc),
    angle_max(_param.cb_sensor_point_angl_max),
    angle_min(_param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns),
    plot(_plot),
    plot_data_segm(_param.viz_data_segm),
    k(_k){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::plot_data(InputData &input, KalmanSLDM k, cv::Scalar color_old, cv::Scalar color_new){
    if(!input.is_valid){
        return;
    }
    plot.putInfoText("[ n]raw_frames no_p = ",input.sensor_raw->size()+input.sensor_raw->size(),1,plot.black);
    plot.putInfoText("[ms]delta t_frame = ",input.u.dt,2,plot.black);
    plot.plot_segm_init(k.seg_init,plot.green_dark);
    plot.plot_segm_init(input.seg_init,plot.green_bright);
    if(plot_data_segm){
        plot.plot_segm(k.seg_ext,plot.blue);
        plot.plot_segm(input.seg_ext,plot.red);
    }

}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::run(InputData &input, KalmanSLDM &k, bool advance){
    if(!k.seg_init){
        if(input.is_valid){
            assign_seg_init(input.sensor_filtered,input.seg_init);
            calc_com(input.seg_init);
        }
        return;
    }
    tf_frm.init(k.rob_x_old(), k.rob_x());

    assign_seg_init(input.sensor_filtered,input.seg_init);
    calc_com(input.seg_init);
    compute_len_init(input.seg_init);


    assign_seg_ext  (input.seg_init , input.seg_ext);
    calc_tf           (input.seg_ext  , NEW2OLD);
    calc_occlusion    (input.seg_ext  , 1      );
//    sample_const_angle(input.seg_ext );
    erase_img_outl    (input.seg_ext  );
    split_segments    (input.seg_ext  );

    calc_tf           (input.seg_ext  , OLD2NEW);
//    sample_const_angle(input.seg_ext);
    calc_com_ext(input.seg_ext);

    inform_init_to_ext(input.seg_init, input.seg_ext, 0);




    sort_seg_init     (k.seg_init);
    compute_len_init  (k.seg_init);




    assign_seg_ext  (k.seg_init , k.seg_ext);//TODO do not include out of view part
    split_segments_for_occl(k.seg_ext  );

    calc_occlusion    (k.seg_ext  , 0      );
    calc_tf           (k.seg_ext  , OLD2NEW);
    calc_occlusion    (k.seg_ext  , 1      );
//    sample_const_angle(k.seg_ext);
    erase_img_outl    (k.seg_ext  );
    split_segments    (k.seg_ext  );
    calc_com_ext(k.seg_ext);

    assign_seg_init_tf(input.seg_init);
    calc_tf_init(k.seg_init, OLD2NEW);

    //TODO have to tf init in t, after that you can do this
    inform_init_to_ext(k.seg_init, k.seg_ext, 1);
}

bool ang_sort_func    (SegmentDataPtr i, SegmentDataPtr j) { return (i->p.front().angle    < j->p.front().angle ); }

void Segmentation::sort_seg_init(SegmentDataPtrVectorPtr &segments_init){
    std::sort(segments_init->begin(), segments_init->end(), ang_sort_func);
}

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

void Segmentation::assign_seg_init_tf(SegmentDataPtrVectorPtr &input){
    IteratorIndexSet_ss iis(input);
    if(iis.status() >= IIS_VALID){
        do{
            (*iis.seg())->p_tf.push_back(*iis.p());
        }while( iis.advance(ALL_SEGM, INC));
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::calc_tf(SegmentDataExtPtrVectorPtr &input, TFmode tf_mode ){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    IteratorIndexSet iis(input);
    if(iis.status() >= IIS_VALID){
        do{
            polar tf_point;
            if     ( tf_mode == OLD2NEW ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.ro2rn( tf_sns.s2r( to_xy( *iis.p() ) ) ) ) );
            }
            else if( tf_mode == NEW2OLD ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.rn2ro( tf_sns.s2r( to_xy( *iis.p() ) ) ) ) );
            }
            normalizeAngle( tf_point.angle );
            if(in_range(tf_point)){
                iis.push_bk(temp,PointData(tf_point));
            }
        }while( iis.advance(ALL_SEGM, INC));
    }
    input = temp;
}
void Segmentation::calc_tf_init(SegmentDataPtrVectorPtr &input, TFmode tf_mode ){
    IteratorIndexSet_ss iis(input);
    if(iis.status() >= IIS_VALID){
        do{
            polar tf_point;
            if     ( tf_mode == OLD2NEW ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.ro2rn( tf_sns.s2r( to_xy( *iis.p() ) ) ) ) );
            }
            else if( tf_mode == NEW2OLD ){
                tf_point = to_polar( tf_sns.r2s( tf_frm.rn2ro( tf_sns.s2r( to_xy( *iis.p() ) ) ) ) );
            }
            normalizeAngle( tf_point.angle );
            if(in_range(tf_point)){
                (*iis.seg())->p_tf.push_back(PointData(tf_point));
            }
        }while( iis.advance(ALL_SEGM, INC));
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///



///------------------------------------------------------------------------------------------------------------------------------------------------///
//TODO MAYBE WORK ONLY ON BIGGER R OCCLUDED AND RUN ON BOTH DIRECTIONS THE FUNCTION
void Segmentation::calc_occlusion(SegmentDataExtPtrVectorPtr &input, int occ_type ){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    IteratorIndexSet iis(input);
    if(iis.status() >= IIS_VALID){
        polar angle_max( *iis.p() );
        iis.push_bk(temp,*iis.p());
        while(iis.advance(ALL_SEGM, INC)){
            iis.advance(ALL_SEGM, DEC);//decrement
            if( angle_max.angle < iis.p()->angle ){
                angle_max = polar( *iis.p() );
            }
            SegmentDataExtPtrVectorIter seg1 = iis.seg();//

            iis.advance(ALL_SEGM, INC);//increment

            if(( occ_type == ONE_SEGM ) && ( seg1 != iis.seg() )){//
                angle_max = polar( *iis.p() );
                seg1 = iis.seg();
            }//

            if( iis.p()->angle < angle_max.angle ){
                polar angle_min = polar( *iis.p() );
                if( iis.p()->r < angle_max.r ){
                    while(( temp->size() > 0 ) || (( temp->size() > 0 ) && ( occ_type == ONE_SEGM ) && ( seg1 == --temp->end() ) ) ){
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
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::split_segments_for_occl(SegmentDataExtPtrVectorPtr &input){
    input->reserve(2 * input->size());
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());

    int id=0;
    for(SegmentDataExtPtrVectorIter seg_inp_n = input->begin(); seg_inp_n != input->end(); seg_inp_n++){
        bool split_segment=false;
        for(SegmentDataExtPtrVectorIter seg_inp_p = seg_inp_n + 1; seg_inp_p != input->end(); seg_inp_p++){
            if((*seg_inp_n)->p.back().angle > (*seg_inp_p)->p.back().angle){
                //split seg_inp_n and put it after seg_inp_p and break and continue
                PointDataVectorIter p = (*seg_inp_n)->p.begin();

                double mid_ang = ((*seg_inp_p)->p.back().angle - (*seg_inp_p)->p.front().angle) / 2.0;
                if((p != (*seg_inp_n)->p.end())&&( p->angle < normalizeAngle(mid_ang) )){
                    temp->push_back(SegmentDataExtPtr(new SegmentDataExt(seg_inp_n, id)));//copy till breaking point

                    while((p != (*seg_inp_n)->p.end())&&( p->angle < normalizeAngle(mid_ang) )){
                        temp->back()->p.push_back(PointData(*p));
                        p++;
                    }
                }
                //appending in input the remaining of the segment after seg_inp_p segment
                if(p != (*seg_inp_n)->p.end()){
                    SegmentDataExtPtrVectorIter inp_append = input->insert(seg_inp_p + 1, SegmentDataExtPtr(new SegmentDataExt(seg_inp_n)));
                    while(p != (*seg_inp_n)->p.end()){
                        (*inp_append)->p.push_back(PointData(*p));
                        p++;
                    }
                }
                split_segment=true;
                break;
            }
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
///DONE!!! CALCULATE SEGMENT LENGTH
void Segmentation::split_segments(SegmentDataExtPtrVectorPtr &input){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    bool split_segment=false;

    IteratorIndexSet2 iis2(input);
    iis2.maj().advance(ALL_SEGM,INC);
    if(iis2.status() >=IIS2_VALID){
        double len = 0;
        do{
            iis2.min().push_bk(temp, *iis2.min().p() ,split_segment);//add the min value in temp, splitting segment or not
            double d_min_maj = 0;
            if( iis2.min().seg() == iis2.maj().seg() ){//if min and maj on same segment
                d_min_maj = diff( *iis2.min().p() , *iis2.maj().p() );
                if(  d_min_maj > segm_discont_dist ){
                    split_segment=true;
                    temp->back()->len = len; len = 0;
                }
                else{
                    len +=d_min_maj;
                }
            }
            else{
                temp->back()->len = len; len = 0;
            }
        }while(iis2.advance(ALL_SEGM,INC));
    }
    input=temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::sample_const_angle(SegmentDataExtPtrVectorPtr &input){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(2*input->size());//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    IteratorIndexSet iis0(input);
    IteratorIndexSet iis1(input);
    if(iis1.status() < IIS_VALID){
        return;
    }
    iis1.advance(ALL_SEGM, INC);
    if(iis1.status() >= IIS_VALID){
        do{
            double d_angle=iis0.p()->angle-iis1.p()->angle;
            normalizeAngle(d_angle);
            /////////////////////////////////////////////////////////////////////////////////////FUNCTION IT
            double sample_pos = ( ( iis0.p()->angle - angle_min ) / angle_inc ) ;
            if( ( sample_pos -(double)((int)sample_pos) ) < 0.01 ){//hard code value weird behaviour
                sample_pos=(int)(sample_pos);
            }
            else{
                sample_pos=(int)(sample_pos+1);
            }
            /////////////////////////////////////////////////////////////////////////////////////
            polar sample( 0 , angle_min + ( angle_inc * sample_pos ) );
            if( iis0.seg() == iis1.seg() ){
                if( diff( *iis0.p() , *iis1.p() ) < segm_discont_dist ){//!!!!!!!!!!!!!!!!!!!!!!!!!NEEDED ESPECIALLY FOR 2 OBJ WITH DIFF VEL FROM ONE OBJ

                    while( sample.angle < iis1.p()->angle -0.0001 ){//hard code value

                        Line l_input = get_line_param( to_xy( *iis0.p() ) , to_xy( *iis1.p() ) );
                        Line l_angle = get_line_param( xy( 0 , 0 ) , to_xy( polar( sensor_range_max , sample.angle ) ) );
                        sample.r = diff( polar( 0 , 0 ) , to_polar( get_line_inters( l_input , l_angle ) ) );///NOT totally efficient

                        iis0.push_bk(temp, PointData(sample));
                        sample = polar( 0 , sample.angle + angle_inc  );
                    }
                }
            }
            else if ( sample.angle < iis0.p()->angle +0.0001 ){//hard code value
                sample.r = iis0.p()->r;
                iis0.push_bk(temp, PointData( sample));
            }
            if ( ( iis1.p() == --input->back()->p.end() ) && ( sample.angle < iis1.p()->angle +0.0001 ) ){//hard code value
                sample.r = iis1.p()->r;
                iis0.push_bk(temp, PointData( polar(sample.r,sample.angle)));
            }
        }while(iis1.advance(ALL_SEGM, INC) && iis0.advance(ALL_SEGM, INC));
    }
    input = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::erase_img_outl(SegmentDataExtPtrVectorPtr &input){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    std::vector<bool> temp_valid;

    IteratorIndexSet iis(input);
    if(iis.status() >= IIS_VALID){
        do{
            iis.push_bk(temp,*iis.p());
            temp_valid.push_back(true);
            check_neigh_p( input, temp , temp_valid ,iis);
        }while( iis.advance(ALL_SEGM, INC));


    }
    iis=IteratorIndexSet(temp,REV);
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

void Segmentation::check_neigh_p(const SegmentDataExtPtrVectorPtr &input,SegmentDataExtPtrVectorPtr &temp,std::vector<bool> &temp_valid,IteratorIndexSet iis){
    std::vector<IteratorIndexSet> neigh_seg;
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
        IteratorIndexSet iis_j = iis;

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
        for(std::vector<IteratorIndexSet>::iterator neigh_seg_it=neigh_seg.begin();neigh_seg_it != neigh_seg.end();neigh_seg_it++){
            if(neigh_seg_it->i() < temp_size){
                check_neigh_p( input , temp , temp_valid ,*neigh_seg_it);
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::calc_com(SegmentDataPtrVectorPtr &input){
    double sample_dist = 0.05;//TODO HARDCODED

    for(SegmentDataPtrVector::iterator inp = input->begin(); inp != input->end(); inp++){
        double rest=0;
        xy p_now,p1_last,p2_last;
        double p_nr = 0;
        xy com;
        if((*inp)->p.size() > 0){
            for(PointDataVectorIter it_in=(*inp)->p.begin();it_in!=--(*inp)->p.end();it_in++){
                p1_last=to_xy(*it_in);
                p2_last=to_xy(*++it_in);
                while(rest+diff(to_polar(p1_last),*it_in) > sample_dist){

                    double k=(sample_dist-rest)/(diff(to_polar(p1_last),*it_in));
                    p_now=xy(p1_last.x+(p2_last.x-p1_last.x)*k,p1_last.y+(p2_last.y-p1_last.y)*k);

                    p1_last=p_now;

                    com += p_now;//com computation
                    p_nr++;

                    rest=0;
                }
                rest+=diff(to_polar(p1_last),*it_in);
                --it_in;
            }
            (*inp)->setCom(xy(com.x / p_nr, com.y / p_nr));
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
//DEBUG ONLY
void Segmentation::calc_com_ext(SegmentDataExtPtrVectorPtr &input){
    double sample_dist = 0.05;//TODO HARDCODED

    for(SegmentDataExtPtrVector::iterator inp = input->begin(); inp != input->end(); inp++){
        double rest=0;
        xy p_now,p1_last,p2_last;
        double p_nr = 0;
        xy com;
        if((*inp)->p.size() > 0){
            for(PointDataVectorIter it_in=(*inp)->p.begin();it_in!=--(*inp)->p.end();it_in++){
                p1_last=to_xy(*it_in);
                p2_last=to_xy(*++it_in);
                while(rest+diff(to_polar(p1_last),*it_in) > sample_dist){

                    double k=(sample_dist-rest)/(diff(to_polar(p1_last),*it_in));
                    p_now=xy(p1_last.x+(p2_last.x-p1_last.x)*k,p1_last.y+(p2_last.y-p1_last.y)*k);

                    p1_last=p_now;

                    com += p_now;//com computation
                    p_nr++;

                    rest=0;
                }
                rest+=diff(to_polar(p1_last),*it_in);
                --it_in;
            }
            (*inp)->setCom(xy(com.x / p_nr, com.y / p_nr));
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
//TODO not really efficient
void Segmentation::inform_init_to_ext(SegmentDataPtrVectorPtr &init, SegmentDataExtPtrVectorPtr &ext, bool old_frame){

    for(SegmentDataExtPtrVectorIter inp = ext->begin(); inp != ext->end(); inp++){

        PointData p_ext[2] = {(*inp)->p.front(), (*inp)->p.back()};

        PointDataVectorIter bound_low;
        PointDataVectorIter bound_high;
        if(old_frame){
            bound_low  = (*inp)->getParrent()->p_tf.begin();
            bound_high = (*inp)->getParrent()->p_tf.end();
        }
        else{
            bound_low  = (*inp)->getParrent()->p.begin();
            bound_high = (*inp)->getParrent()->p.end();
        }
        double d_min[2] = {10000, 10000};
        int p=0;
        int p_min[2] = {0,0};
        for(PointDataVectorIter it_in = bound_low;it_in != bound_high;it_in++){
            if(diff(*it_in, p_ext[0]) < d_min[0]){
                d_min[0] = diff(*it_in, p_ext[0]);
                p_min[0] = p;
            }
            if(diff(*it_in, p_ext[1]) < d_min[1]){
                d_min[1] = diff(*it_in, p_ext[1]);
                p_min[1] = p;
            }
            p++;
        }
//        bound_low  = (*inp)->getParrent()->p.begin();
        for(PointDataVectorIter it_in = bound_low + p_min[0];(it_in != bound_low + p_min[1] + 1);it_in++){
            it_in->child = (*inp);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::compute_len_init(SegmentDataPtrVectorPtr &input){
    SegmentDataPtrVectorPtr temp( new SegmentDataPtrVector );
    temp->reserve(input->size());
    bool split_segment=false;

    IteratorIndexSet2_ss iis2(input);
    iis2.maj().advance(ALL_SEGM,INC);
    if(iis2.status() >=IIS2_VALID){
        double len = 0;
        do{
            iis2.min().push_bk(temp, *iis2.min().p() ,split_segment);//add the min value in temp, splitting segment or not
            if( iis2.min().seg() == iis2.maj().seg() ){//if min and maj on same segment
                len += diff( *iis2.min().p() , *iis2.maj().p() );
            }
            else{
                temp->back()->len = len; len = 0;
            }
        }while(iis2.advance(ALL_SEGM,INC));
    }
    input=temp;
}
