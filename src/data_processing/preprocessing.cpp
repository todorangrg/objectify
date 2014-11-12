#include "data_processing/preprocessing.h"
#include "utils/math.h"
#include "visual/plot_data.h"


Preprocessing::Preprocessing(RecfgParam &_param, PlotData& _plot):
    filter_input(_param.preproc_filter),
    filter_circle_rad(_param.preproc_filter_circle_rad),
    filter_circle_rad_scale(_param.preproc_filter_circle_rad_scale),
    filter_sigma(_param.preproc_filter_sigma),
    outl_circle_rad(_param.segm_outl_circle_rad),
    outl_sigma(_param.segm_outl_sigma),
    outl_prob_thres(_param.segm_outl_prob_thres),
    angle_inc(_param.cb_sensor_point_angl_inc),
    plot(_plot),
    plot_data_raw(_param.viz_data_raw),
    plot_data_preproc(_param.viz_data_preproc),
    plot_oult_preproc(_param.viz_data_oult_preproc){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Preprocessing::plot_data(SensorData& sensor, cv::Scalar color_raw, cv::Scalar color_preproc, cv::Scalar color_outl_acc,cv::Scalar color_outl_rej){
    if(sensor.status == NO_FRAME){
        return;
    }
    if(plot_data_raw){
        plot.plot_points(sensor.frame_new->sensor_raw,color_raw);
    }
    if(plot_data_preproc){
        plot.plot_points(sensor.frame_new->sensor_filtered,color_preproc);
    }
    if(plot_oult_preproc){
        plot.plot_oult_circles(plot_p_ell,plot_p_ell_erased,color_outl_acc,color_outl_rej);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Preprocessing::run(SensorData& sensor){
    if(sensor.status == NO_FRAME){
        return;
    }
    if( filter_input == true ){
        filter(sensor.frame_new->sensor_raw, sensor.frame_new->sensor_filtered);
        erase_outl(sensor.frame_new->sensor_filtered, sensor.frame_new->sensor_filtered, plot_oult_preproc);
    }
    else{
        erase_outl(sensor.frame_new->sensor_raw, sensor.frame_new->sensor_filtered, plot_oult_preproc);
    }   
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Preprocessing::filter( const PointDataVectorPtr &input , PointDataVectorPtr &output ){
    PointDataVectorPtr temp( new PointDataVector );
    temp->reserve(input->size());
    for( int i=0 ; i < input->size() ; i++ ){
        temp->push_back( input->at(i) );
        double circle_rad;
        if( filter_circle_rad_scale == false ){
            circle_rad = filter_circle_rad;
        }
        else{
            circle_rad = sqrt( 2.0 * sqr( input->at(i).r ) * ( 1.0 - cos( angle_inc )  ) ) * filter_circle_rad / sqrt( 2.0 * ( 1.0 - cos( angle_inc )  ) );
        }
        long double p_prob[2] = {0}; //for [0] negative( [1] positive) - to the right(to the left) neighbouring points
        long double p_r_avg[2] = {0};//for [0] negative( [1] positive) - to the right(to the left) neighbouring points
        polar p_i( input->at(i) );
        double search_angle;
        if( fabs( circle_rad / p_i.r ) > 1.0  ){
            search_angle = 2 * M_PI;
        }
        else{
            search_angle = asin( circle_rad / p_i.r );
        }
        for( int dir = -1 ; dir < 2 ; dir += 2 ){
            for( int j = i + dir ; ( ( j >= 0 ) && ( (*input)[j].angle >= p_i.angle - search_angle ) ) && ( ( j < input->size() ) && ( (*input)[j].angle <= p_i.angle + search_angle ) ) ; j += dir ){
                polar p_j = polar( input->at(j) );
                if( diff( p_j , p_i ) <= circle_rad ){
                    long double r_j_t = p_j.r * cos( p_j.angle - p_i.angle );
                    long double prob_j = std::exp( - ( sqr( diff( p_j , p_i ) ) ) / ( filter_sigma * sqr( circle_rad ) ) );
                    p_r_avg[std::max( dir , 0 )] += prob_j * r_j_t;
                    p_prob [std::max( dir , 0 )] += prob_j;
                }
            }
        }
        if( p_prob[0] < p_prob[1] ){
            p_r_avg[1] *= p_prob[0] / p_prob[1];
            p_prob [1]  = p_prob[0];
        }
        else if( p_prob[0] > p_prob[1] ){
            p_r_avg[0] *= p_prob[1] / p_prob[0];
            p_prob[0]   = p_prob[1];
        }
        temp->back().r = ( p_r_avg[0] + p_r_avg[1] + input->at(i).r ) / ( 1.0 + p_prob[0] + p_prob[1] );
    }
    output = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Preprocessing::erase_outl( const PointDataVectorPtr &input , PointDataVectorPtr &output , bool debug_print ){
    PointDataVectorPtr temp( new PointDataVector );
    temp->reserve(input->size());
    std::vector<bool> temp_valid;

    if( debug_print ){
        plot_p_ell.clear();
        plot_p_ell_erased.clear();
    }
    for( int i=0 ; i < input->size() ; i++ ){
        temp->push_back( input->at(i) );
        temp_valid.push_back(true);
        if( debug_print ){
            plot_p_ell.push_back( cv::RotatedRect( to_xy( input->at(i) ) , cv::Size2f( outl_circle_rad , outl_circle_rad ) , -rad_to_deg( input->at(i).angle ) ) );
            plot_p_ell_erased.push_back( 0 );//normal value
        }
        check_neigh_p( input, temp , temp_valid , i , debug_print );
    }
    for(int i = temp_valid.size() - 1 ; i >= 0 ; i-- ){
        if( !temp_valid[i] ){
            temp->erase( temp->begin() + i );
        }
    }
    output = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Preprocessing::check_neigh_p( const PointDataVectorPtr &input , PointDataVectorPtr &temp , std::vector<bool> &temp_valid , int it , bool debug_print ){
    std::vector<int> neigh_p;
    double p_prob = 0;
    polar p_i( input->at(it) );
    double search_angle;
    if( fabs( outl_circle_rad / p_i.r ) > 1.0  ){
        search_angle = 2 * M_PI;
    }
    else{
        search_angle = asin( outl_circle_rad / p_i.r );
    }
    for( int dir = -1 ; dir < 2 ; dir += 2 ){
        for( int j = it + dir ; ( ( j >= 0 ) && ( (*input)[j].angle >= p_i.angle - search_angle ) ) && ( ( j < input->size() ) && ( (*input)[j].angle <= p_i.angle + search_angle ) ) ; j += dir ){
            if( ( temp_valid[j] == true ) || ( j >= temp_valid.size() ) ){
                polar p_j = polar( input->at(j) );
                neigh_p.push_back( j );
                if( diff( p_j , p_i ) <= outl_circle_rad ){
                    p_prob  += std::exp( - ( sqr( diff( p_j , p_i ) ) ) / ( outl_sigma * sqr( outl_circle_rad ) ) );
                    if( p_prob >= outl_prob_thres ){
                        break;
                    }
                }
            }
        }
    }
    if( p_prob < outl_prob_thres ){
        temp_valid[ it ] = false;
        if( debug_print ){
            plot_p_ell_erased.at( it ) = 1 ;//erased value
        }
        for( int k = 0 ; k < neigh_p.size() ; k++ ){
            if( neigh_p[k] < temp->size() ){
                check_neigh_p( input , temp , temp_valid , neigh_p[k] , debug_print );
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
