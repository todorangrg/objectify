
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "utils/iterators.h"
#include "utils/base_classes.h"
#include "data_processing/dp.h"

#include "utils/convolution.h"
#include "visual/plot_data.h"


using namespace cv;


PlotData::PlotData(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns):
    Plot(wndView),
    plot_data(_param.viz_data),
    plot_grid(_param.viz_data_grid),
    image_size(_param.viz_data_res),
    sensor_range_max(_param.sensor_range_max),
    angle_max(_param.cb_sensor_point_angl_max),
    angle_min(_param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns){

    namedWindow(wndView_,CV_GUI_EXPANDED);
    resizeWindow(wndView_,1300,900);
    cv::moveWindow(wndView_, 0, 0);
    plot = Mat(900, 900 + 400, CV_8UC3);
    init_w2i();
    plot(Range(0,900),Range(0,900)).setTo(white);
    plot(Range(0,900),Range(900,900 + 400)).setTo(gray);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_points(const PointDataVectorPtr &data,Scalar color){
    for (unsigned int i = 0; i < data->size(); i++){
        circle(plot, w2i(tf_sns.s2r(to_xy(data->at(i)))),2,color);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_oult_circles(const std::vector<cv::RotatedRect>& plot_p_ell, std::vector<int>& plot_p_ell_erased, cv::Scalar color_acc,cv::Scalar color_rej){
    for(int i=0;i<plot_p_ell.size();i++){
        RotatedRect plot_ellipse;
        plot_ellipse.center=w2i(tf_sns.s2r(plot_p_ell[i].center));
        plot_ellipse.size.width=Mw2i[2][2]*2*plot_p_ell[i].size.width-3;
        plot_ellipse.size.height=Mw2i[2][2]*2*plot_p_ell[i].size.height-3;
        plot_ellipse.angle=plot_p_ell[i].angle;
        if(plot_p_ell_erased[i]==0){
            ellipse(plot,plot_ellipse, color_acc, 1, CV_AA);
        }
        else if(plot_p_ell_erased[i]==1){
            ellipse(plot,plot_ellipse, color_rej, 1, CV_AA);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_segm(const SegmentDataExtPtrVectorPtr &data,cv::Scalar color){
    IteratorIndexSet iis(data);
    if(iis.status() >= IIS_VALID){
        do{
            circle(plot, w2i(tf_sns.s2r(to_xy(*iis.p()))),2,color);
//            line(plot,w2i(s2r(0,0)), w2i(s2r(to_xy(*iis.p))),blue);
            if( iis.status() == IIS_P_RBEGIN ){
                putFullCircle(w2i(tf_sns.s2r((*iis.seg())->conv->com)),1,5,color);
                std::stringstream s;
                s.precision(4);
                s << /*"len="<<std::setw(3)<<(*iis.seg())->len<<*/ "  S:"<<(*iis.seg())->id;
                if((*iis.seg())->parrent->parrent->fr_stat == FRAME_OLD){
                    s <<"|O:"<<(*iis.seg())->parrent->parrent->id;
                }
                putText(plot,s.str().c_str(),w2i(tf_sns.s2r((*iis.seg())->conv->com)),FONT_HERSHEY_PLAIN,1,color);
            }
        }while( iis.advance(ALL_SEGM, INC));
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_segm_tf(const SegmentDataExtPtrVectorPtr &data, int frame, cv::Scalar color){
    IteratorIndexSet iis(data);
    if(iis.status() >= IIS_VALID){
        do{
            TfVar tf;
            for(std::vector<TFdata>::iterator tf_it = (*iis.seg())->conv->tf->begin(); tf_it != (*iis.seg())->conv->tf->end(); tf_it++){
                if(tf_it->conv_stat == CONV_SPL ){
                    continue;
                }
                xy com_new, com_old, p_first_old,p_first_new, p_last_old,p_last_new;

                if( iis.status() == IIS_P_RBEGIN ){
                    if( (*iis.seg())->parrent->parrent->fr_stat == FRAME_OLD ){
                        tf = tf_it->tf;
                        com_old = (*iis.seg())->conv->com;
                        p_first_old = to_xy((*iis.seg())->p.front());
                        p_last_old  = to_xy((*iis.seg())->p.back());
                        com_new     = mat_mult(tf.T, com_old);
                        p_first_new = mat_mult(tf.T, p_first_old);
                        p_last_new  = mat_mult(tf.T, p_last_old);
                    }
                    else{
                        tf = tf_it->tf_inv;
                        com_new = (*iis.seg())->conv->com;
                        p_first_new = to_xy((*iis.seg())->p.front());
                        p_last_new  = to_xy((*iis.seg())->p.back());
                        com_old     = mat_mult(tf_it->tf.T, com_new);
                        p_first_old = mat_mult(tf_it->tf.T, p_first_new);
                        p_last_old  = mat_mult(tf_it->tf.T, p_last_new);
                    }

                    putFullCircle(w2i(tf_sns.s2r(com_new)),1,5,black);

                    xy arrow_base = w2i(tf_sns.s2r(com_old));
                    xy arrow_tip  = w2i(tf_sns.s2r(com_old + tf.xy_mean));
                    putArrow(arrow_base, arrow_tip,green_dark,2);
                    arrow_base = w2i(tf_sns.s2r(p_first_old));
                    arrow_tip  = w2i(tf_sns.s2r(p_first_new));
                    putArrow(arrow_base, arrow_tip,green_dark,2);
                    arrow_base = w2i(tf_sns.s2r(p_last_old));
                    arrow_tip  = w2i(tf_sns.s2r(p_last_new));
                    putArrow(arrow_base, arrow_tip,green_dark,2);


                    cv::Matx33d cov_xy33(tf.xy_cov(0,0), tf.xy_cov(0,1), 0,
                                         tf.xy_cov(1,0), tf.xy_cov(1,1), 0,
                                                     0,               0, 0);
                    cv::Matx33d Mw2i33(Mw2i);
                    cov_xy33 = Mw2i33 * cov_xy33 * Mw2i33.t();
                    cv::Matx22d cov_xy22(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
                    cv::RotatedRect ellips = cov2rect(cov_xy22,w2i(tf_sns.s2r(com_old + tf.xy_mean)));
                    cv::ellipse(plot,ellips,green_bright,2);
                }
                xy p_tf;
                if( (*iis.seg())->parrent->parrent->fr_stat == FRAME_OLD ){
                    p_tf = mat_mult(tf_it->tf.T,to_xy(*iis.p()));
                }
                else{
                    p_tf = to_xy(*iis.p());
                }
                circle(plot, w2i(tf_sns.s2r(p_tf)), 2, black);
            }
        }while( iis.advance(ALL_SEGM, INC));
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_kalman(const SegmentDataExtPtrVectorPtr &data, KalmanSLDM& k){
    putArrow(w2i(0,0),w2i(k.S.at<double>(0,0) - k.S_R_bar.at<double>(0,0),k.S.at<double>(1,0) - k.S_R_bar.at<double>(1,0)),magenta,2);

    cv::Matx33d Mw2i33(Mw2i);
    cv::Matx22d Mw2i22(Mw2i33(0,0),Mw2i33(0,1),Mw2i33(1,0),Mw2i33(1,1));
    cv::Matx33d cov_xy33(k.P.rowRange(0,3).colRange(0,3));

    cv::Matx22d cov_xy22(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));

    double rob_bar_xx_f0   = k.S_R_bar.at<double>(0,0);
    double rob_bar_xy_f0   = k.S_R_bar.at<double>(1,0);
    double rob_bar_xphi_f0 = k.S_R_bar.at<double>(2,0);
    cv::Matx22d rot_rob_bar(cos(-rob_bar_xphi_f0),-sin(-rob_bar_xphi_f0),sin(-rob_bar_xphi_f0), cos(-rob_bar_xphi_f0));

    cov_xy22 = Mw2i22 * rot_rob_bar * cov_xy22 *rot_rob_bar.t() * Mw2i22.t();



    cv::RotatedRect ellips = cov2rect(cov_xy22,w2i(k.S.at<double>(0,0) - rob_bar_xx_f0,k.S.at<double>(1,0) - rob_bar_xy_f0));
    cv::ellipse(plot,ellips,magenta,2);

    for(int i=0; i< data->size(); i++){
        if( data->at(i)->conv ){
            if(data->at(i)->conv->tf->size() > 0){
                if(data->at(i)->conv->tf->front().seg->parrent){
                    xy com  =  data->at(i)->conv->com;
                    ObjectDataPtr obj = data->at(i)->conv->tf->front().seg->parrent->parrent;
                    if( k.Oi.count(obj) != 0 ){
                        xy v(k.Oi[obj].S_O.at<double>(3,0),k.Oi[obj].S_O.at<double>(4,0));double w = k.Oi[obj].S_O.at<double>(5,0);
                        v = rot_rob_bar * v;
                        putArrow(w2i(tf_sns.s2r(com)),w2i(tf_sns.s2r(com + xy(v.x,v.y))),magenta,2);

                        Mw2i22 = cv::Matx22d(Mw2i33(0,0),Mw2i33(0,1),Mw2i33(1,0),Mw2i33(1,1));
                        cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(3,6).colRange(3,6));

                        cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));

                        cov_xy22 = Mw2i22 * rot_rob_bar * cov_xy22 * rot_rob_bar.t() * Mw2i22.t();

                        ellips = cov2rect(cov_xy22,w2i(tf_sns.s2r(com + xy(v.x,v.y))));
                        cv::ellipse(plot,ellips,magenta,2);
                    }
                }
            }
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_corr_links(const std::vector<CorrInput>& list,cv::Scalar color_o2n,cv::Scalar color_n2o){
    for(std::vector<CorrInput>::const_iterator it_corr=list.begin();it_corr!=list.end();it_corr++){
        if( it_corr->reverse ){
            putArrow(w2i(tf_sns.s2r(it_corr->frame_new->conv->com)),w2i(tf_sns.s2r(it_corr->frame_old->conv->com)),color_n2o,2);
        }
        else{
            putArrow(w2i(tf_sns.s2r(it_corr->frame_old->conv->com)),w2i(tf_sns.s2r(it_corr->frame_new->conv->com)),color_o2n,2);
        }
        std::stringstream ss; ss << 100*it_corr->stitch_perc<<"%";
        putText(plot,ss.str().c_str(),w2i(tf_sns.s2r(xy((it_corr->frame_old->conv->com.x+it_corr->frame_new->conv->com.x)/2.0,
                                                        (it_corr->frame_old->conv->com.y+it_corr->frame_new->conv->com.y)/2.0))),FONT_HERSHEY_PLAIN,0.5,black);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::update(){
    if(plot_data){
        imshow(wndView_,plot);
        waitKey(10);

        plot.release();
        namedWindow(wndView_, 1);
//        cv::moveWindow(wndView_, 0, 0);
        plot = Mat(image_size, image_size + 400, CV_8UC3);
        init_w2i();
        plot(Range(0,image_size),Range(0,image_size)).setTo(white);
        plot(Range(0,image_size),Range(image_size,image_size + 400)).setTo(gray);

        if (plot_grid) {
            for (double y = -round(sensor_range_max + 1); y <= round(sensor_range_max + 1); y+=1.0) {
                for (double x = -round(sensor_range_max + 1); x <= round(sensor_range_max + 1); x+=1.0) {
                    circle(plot, w2i(x,y), 1, black);
                }
            }
            line(plot, w2i(-round(sensor_range_max + 1),0), w2i(round(sensor_range_max + 1),0), black);
            line(plot, w2i(0,-round(sensor_range_max + 1)), w2i(0,round(sensor_range_max + 1)), black);
            putText(plot, "X [5m]", w2i(5.2, -0.3), FONT_HERSHEY_PLAIN, 1, black);
            putText(plot, "Y [5m]", w2i(0.2, 5.5), FONT_HERSHEY_PLAIN, 1, black);

            ellipse(plot, w2i(tf_sns.s2r(0,0)), Size( Mw2i[2][2] * sensor_range_max, Mw2i[2][2]* sensor_range_max), rad_to_deg(-acos(Mw2i[0][0])), rad_to_deg(angle_min), rad_to_deg(angle_max), black );
            line(plot, w2i(tf_sns.s2r(0,0)), w2i(tf_sns.s2r(sensor_range_max * cos(angle_max), sensor_range_max * sin(angle_max))), black);
            line(plot, w2i(tf_sns.s2r(0,0)), w2i(tf_sns.s2r(sensor_range_max * cos(angle_min), sensor_range_max * sin(angle_min))), black);
        }

    }
    else{
        destroyWindow(wndView_);
        waitKey(10);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::putInfoText(const string &text,double var,int row,Scalar color){
    std::stringstream ss;
    ss << var;
    putText(plot,text+ss.str().c_str(),xy(image_size,row*15),FONT_HERSHEY_PLAIN,1,color);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::init_w2i() {
    double x_y_d = round(sensor_range_max) + 1;
    int scale = (double)image_size / ( x_y_d * 2.0 );
    Mat_<double> Sc = (Mat_<double>(3,3) << scale, 0, 0,
                                            0, scale, 0,
                                            0, 0, scale);   // Scale
    Mat_<double> M = (Mat_<double>(3,3) << -1, 0, 0,
                                           0, 1, 0,
                                           0, 0, 1);    // Mirror
    Mat_<double> R = (Mat_<double>(3,3) << 0, -1, 0,
                                           1, 0, 0,
                                           0, 0, 1);    // Rotate
    Mat_<double> T = (Mat_<double>(3,3) << 1, 0, x_y_d,
                                           0, 1, x_y_d,
                                           0, 0, 1);    // Translate

    // Calculate final transformation matrix
    Mw2i = T * R * M * Sc;
//    std::cout <<  "Mw2i = " << std::endl << Mw2i << std::endl;
}

