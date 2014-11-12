#include "data_processing/segmentation.h"
#include "utils/math.h"
#include "utils/iterators.h"
#include "utils/kalman.h"

extern const char* ConvFlagNames[];


//TODO sort the recieved old-frames to be still angular sorted


Segmentation::Segmentation(RecfgParam &_param, SensorTf& _tf_sns, FrameTf& _tf_frm, PlotData& _plot, KalmanSLDM &_k) :
    outl_circle_rad(_param.segm_outl_circle_rad),
    outl_sigma(_param.segm_outl_sigma),
    outl_prob_thres(_param.segm_outl_prob_thres),
    segm_discont_dist(_param.segm_discont_dist),
    sensor_range_max(_param.sensor_range_max),
    angle_inc(_param.cb_sensor_point_angl_inc),
    angle_max(_param.cb_sensor_point_angl_max),
    angle_min(_param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns),
    tf_frm(_tf_frm),
    plot(_plot),
    plot_data_segm(_param.viz_data_segm),
    k(_k){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::plot_data(SensorData& sensor, cv::Scalar color_old, cv::Scalar color_new){
    if(sensor.status != OLD_FRAME){
        return;
    }
    plot.putInfoText("[ n]raw_frames no_p = ",sensor.frame_old->sensor_raw->size()+sensor.frame_new->sensor_raw->size(),1,plot.black);
    plot.putInfoText("[ms]delta t_frame = ",sensor.frame_old->past_time.toNSec()* 1e-6 ,2,plot.black);
    if(plot_data_segm){
        plot.plot_segm(sensor.frame_old->seg_ext,plot.blue);
        plot.plot_segm(sensor.frame_new->seg_ext,plot.red);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::run(SensorData& sensor, bool advance){
    if(sensor.status != OLD_FRAME){
        if(sensor.status == NEW_FRAME){
            assign_to_seg_init(sensor.frame_new->sensor_filtered,sensor.frame_new->seg_init, sensor.objects);
            calc_com(sensor.frame_new->seg_init);
        }
        return;
    }
    assign_to_seg_init(sensor.frame_new->sensor_filtered,sensor.frame_new->seg_init, sensor.objects);
    calc_com(sensor.frame_new->seg_init);

    populate_seg_ext  (sensor.frame_new->seg_init , sensor.frame_new->seg_ext);
    calc_tf           (sensor.frame_new->seg_ext  , NEW2OLD);
    calc_occlusion    (sensor.frame_new->seg_ext  , 1      );
//    sample_const_angle(sensor.frame_new->seg_ext  );
    erase_img_outl    (sensor.frame_new->seg_ext  );
    split_segments    (sensor.frame_new->seg_ext  );

    calc_tf           (sensor.frame_new->seg_ext  , OLD2NEW);
//    sample_const_angle(sensor.frame_new->seg_ext  );

    if(advance){
        calc_tf_vel       (sensor.frame_old->seg_init, sensor);
    }
    sort_seg_init     (sensor.frame_old->seg_init);
    populate_seg_ext  (sensor.frame_old->seg_init , sensor.frame_old->seg_ext);

    calc_occlusion    (sensor.frame_old->seg_ext  , 0      );
    calc_tf           (sensor.frame_old->seg_ext  , OLD2NEW);
    calc_occlusion    (sensor.frame_old->seg_ext  , 1      );
//    sample_const_angle(sensor.frame_old->seg_ext  );
    erase_img_outl    (sensor.frame_old->seg_ext  );
    split_segments    (sensor.frame_old->seg_ext  );
}

bool ang_sort_func    (SegmentDataPtr i, SegmentDataPtr j) { return (i->p.front().angle    < j->p.front().angle ); }

void Segmentation::sort_seg_init(SegmentDataPtrVectorPtr &segments_init){
    std::sort(segments_init->begin(), segments_init->end(), ang_sort_func);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::assign_to_seg_init(const PointDataVectorPtr &input , SegmentDataPtrVectorPtr &segments_init, boost::shared_ptr<std::vector<boost::shared_ptr<ObjectData> > > obj){
    segments_init->clear();

    obj->push_back(boost::shared_ptr<ObjectData>(new ObjectData(FRAME_NEW, 0)));
    segments_init->push_back(SegmentDataPtr(new SegmentData(obj->back(),0)));
    xy com = xy (0,0);
    double com_k=0.0;
    if(input->size() >0){
        for( int i = 1 ; i < input->size() ; i++ ){
            segments_init->back()->p.push_back(input->at(i-1));
            com += to_xy( input->at(i-1) );com_k+=1.0;
            if ( diff( input->at(i-1) , input->at(i) ) > segm_discont_dist ){
                com = xy( 0.0,0.0 );com_k=0.0;
                obj->push_back(boost::shared_ptr<ObjectData>(new ObjectData(FRAME_NEW, obj->back()->id+1)));
                segments_init->push_back(SegmentDataPtr(new SegmentData(obj->back(),segments_init->back()->id+1)));
            }
        }
        com += to_xy( input->back() );com_k+=1.0;
        segments_init->back()->p.push_back(input->back());
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::populate_seg_ext(const SegmentDataPtrVectorPtr &input, SegmentDataExtPtrVectorPtr &output){
    SegmentDataExtPtrVectorPtr temp( new SegmentDataExtPtrVector );
    temp->reserve(input->size());
    for(SegmentDataPtrVectorIter it_in = input->begin(); it_in != input->end() ; it_in++){
        temp->push_back(SegmentDataExtPtr(new SegmentDataExt(*it_in)));
        for(PointDataVectorIter it_p = (*it_in)->p.begin(); it_p != (*it_in)->p.end(); it_p++){
            temp->back()->p.push_back(*it_p);
        }
    }
    output = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Segmentation::calc_tf_vel(SegmentDataPtrVectorPtr &input, SensorData& sensor){///!!!!!SOME ISSUES HERE, AT HIGH ACCEL OF R1 WITH SMALL OBJ VEL THE TF IS BIG
    for(SegmentDataPtrVector::iterator inp = input->begin(); inp != input->end(); inp++){
        if(k.Oi.count((*inp)->parrent) == 0){
            continue;
        }
        double dt = sensor.frame_old->past_time.toSec();//ALL THIS TF HAS TO BE TF-ED ITSELF IN NEW ROB_BAR FRAME
        cv::Mat S_O = k.Oi[(*inp)->parrent].S_O;
        RState  rob_bar_f0;
        rob_bar_f0.xx   = sensor.frame_old->rob_x.x; rob_bar_f0.xy   = sensor.frame_old->rob_x.y; rob_bar_f0.xphi = sensor.frame_old->rob_x.angle;
        OiState obj_f0(S_O);
        OiState obj_f1;

        obj_f1.xx   =  (obj_f0.xx - rob_bar_f0.xx) * cos(rob_bar_f0.xphi) + (obj_f0.xy - rob_bar_f0.xy) * sin(rob_bar_f0.xphi);
        obj_f1.xy   =  (obj_f0.xy - rob_bar_f0.xy) * cos(rob_bar_f0.xphi) - (obj_f0.xx - rob_bar_f0.xx) * sin(rob_bar_f0.xphi);
        obj_f1.xphi =   obj_f0.xphi - rob_bar_f0.xphi;
        obj_f1.vx   =   obj_f0.vx * cos(rob_bar_f0.xphi) + obj_f0.vy * sin(rob_bar_f0.xphi);
        obj_f1.vy   =   obj_f0.vy * cos(rob_bar_f0.xphi) - obj_f0.vx * sin(rob_bar_f0.xphi);
        obj_f1.vphi =   obj_f0.vphi;
        obj_f1.ax   =   obj_f0.ax * cos(rob_bar_f0.xphi) + obj_f0.ay * sin(rob_bar_f0.xphi);
        obj_f1.ay   =   obj_f0.ay * cos(rob_bar_f0.xphi) - obj_f0.ax * sin(rob_bar_f0.xphi);
        obj_f1.aphi =   obj_f0.aphi;

        xy com_x_f1(obj_f1.xx/*(*inp)->parrent->com.x*/, obj_f1.xy/*(*inp)->parrent->com.y*/);

        xy t;double angle;
        t.x = obj_f1.vx * dt + obj_f1.ax * sqr(dt) / 2.0;
        t.y = obj_f1.vy * dt + obj_f1.ay * sqr(dt) / 2.0;
        angle = /*obj_xphi_f1 + */(obj_f1.vphi * dt + obj_f1.aphi * sqr(dt) / 2.0);

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
            if( ( iis.p()->r < sensor_range_max ) && ( tf_point.r < sensor_range_max ) && ( tf_point.angle <= angle_max ) && ( tf_point.angle >= angle_min ) ){
                iis.push_bk(temp,PointData(tf_point/*,iis.p()->id*/));
            }
        }while( iis.advance(ALL_SEGM, INC));
    }
    input = temp;
}

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

    double rest=0;
    xy p_now,p1_last,p2_last;
    double p_nr = 0;

    xy com;
    for(SegmentDataPtrVector::iterator inp = input->begin(); inp != input->end(); inp++){
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
            (*inp)->com = xy(com.x / p_nr, com.y / p_nr);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
