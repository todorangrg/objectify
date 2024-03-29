/***************************************************************************
 *   Software License Agreement (BSD License)                              *
 *   Copyright (C) 2015 by Horatiu George Todoran <todorangrg@gmail.com>   *
 *                                                                         *
 *   Redistribution and use in source and binary forms, with or without    *
 *   modification, are permitted provided that the following conditions    *
 *   are met:                                                              *
 *                                                                         *
 *   1. Redistributions of source code must retain the above copyright     *
 *      notice, this list of conditions and the following disclaimer.      *
 *   2. Redistributions in binary form must reproduce the above copyright  *
 *      notice, this list of conditions and the following disclaimer in    *
 *      the documentation and/or other materials provided with the         *
 *      distribution.                                                      *
 *   3. Neither the name of the copyright holder nor the names of its      *
 *      contributors may be used to endorse or promote products derived    *
 *      from this software without specific prior written permission.      *
 *                                                                         *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   *
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     *
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS     *
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE        *
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,  *
 *   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
 *   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      *
 *   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      *
 *   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT    *
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY *
 *   WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           *
 *   POSSIBILITY OF SUCH DAMAGE.                                           *
 ***************************************************************************/


#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "utils/iterators.h"
#include "utils/base_classes.h"
#include "data_processing/dp.h"

#include "utils/convolution.h"
#include "visual/plot_data.h"

#include "boost/type_traits/is_same.hpp"

using namespace cv;

PlotData::PlotData(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns):
    Plot(wndView),
    plot_data(_param.viz_data),
    plot_grid(_param.viz_data_grid),
    sensor_range_max(_param.sensor_r_max),
    angle_max(_param.cb_sensor_point_angl_max),
    angle_min(_param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns),
    image_size(900){

    namedWindow(wndView_,CV_GUI_EXPANDED);
    resizeWindow(wndView_,900 + 800,900);
    cv::moveWindow(wndView_, 0, 0);
    plot = Mat(900, 900 + 800, CV_8UC3);
    init_w2i();
    plot(Range(0,900),Range(0,900)).setTo(white);
    plot(Range(0,900),Range(900,900 + 800)).setTo(gray);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_points(PointDataVector &data,Scalar color){
    for (PointDataVectorIter p = data.begin(); p != data.end(); p++){
        circle(plot, w2i(tf_sns.s2r(to_xy(*p))),2,color);
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
template <class SegData>
void PlotData::plot_segm(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data,cv::Scalar color, int name){
    if(!data){
        return;
    }
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator it_data=data->begin(); it_data != data->end(); it_data++){

        plot_points((*it_data)->p, color);

        xy com = (*it_data)->getCom();
        putFullCircle(w2i(tf_sns.s2r(com)),1,4,color);

        if(name != NONE){
            std::stringstream s; s.precision(4);
            //s<<"l:" <<std::setw(3)<< (*it_data)->getLen()
            if(name == ALL){
                s<<"|S:"<<               (*it_data)->id;
            }


            bool frame_old = true, init = true;

            double len = 0.5;
            if(name == ALL){
                s<<"|O:";
                if((*it_data)->getObj()){                        s << (*it_data)->getObj()->id << "(t-1)"; frame_old = true; }
                else{                                            s << "?"                      << "( t )"; frame_old = false;}
                if(boost::is_same<SegData, SegmentData>::value){ s << "i";                                 init      = true; }
                else{                                            s << "e";                                 init      = false;}
            }
            else{
                s<<"|Obj:";
                if((*it_data)->getObj()){                        s << (*it_data)->getObj()->id;}
                else{                                            s << "?"                     ;}
                len = 1.0;
            }
            xy direct;
            ;
            double ang = M_PI / 6.0 ;
            if     (frame_old && init){
                direct = xy(len * cos(4.0 * ang + M_PI / 2.0), len * sin(4.0 * ang + M_PI / 2.0));
            }
            else if(frame_old && !init){
                direct = xy(len * cos( 5.0 * ang + M_PI / 2.0), len * sin( 5.0 * ang + M_PI / 2.0));
            }
            else if(!frame_old && init){
                direct = xy(len * cos(-2.0 * ang + M_PI / 2.0), len * sin(-2.0 * ang + M_PI / 2.0));
            }
            else if(!frame_old && !init){
                direct = xy(len * cos(-1.0 * ang + M_PI / 2.0), len * sin(-1.0 * ang + M_PI / 2.0));
            }
            cv::Scalar color_info = color;
            double font_size = 1;
            if(name == OBJ){
                color_info = black;
                font_size = 1.6;
            }
            line   (plot, w2i(tf_sns.s2r(com)), w2i(tf_sns.s2r(com + direct)), color_info);
            putText(plot,s.str().c_str(),w2i(tf_sns.s2r(com + direct)),FONT_HERSHEY_PLAIN,font_size,color_info);
        }
    }
}

template void PlotData::plot_segm<SegmentData>   (SegmentDataPtrVectorPtr    &data, cv::Scalar color, int name);
template void PlotData::plot_segm<SegmentDataExt>(SegmentDataExtPtrVectorPtr &data, cv::Scalar color, int name);

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_segm_tf(const SegmentDataExtPtrVectorPtr &data, int frame, cv::Scalar color){
//    IteratorIndexSet iis(data);
//    if(iis.status() >= IIS_VALID){
//        do{
//            TfVar tf;
//            for(std::vector<TFdata>::iterator tf_it = (*iis.seg())->conv->tf->begin(); tf_it != (*iis.seg())->conv->tf->end(); tf_it++){
//                if(tf_it->conv_stat == CONV_SPL ){
//                    continue;
//                }
//                xy com_new, com_old, p_first_old,p_first_new, p_last_old,p_last_new;

//                if( iis.status() == IIS_P_RBEGIN ){
//                    if( (*iis.seg())->getObj()){
//                        tf = tf_it->tf;
//                        com_old = (*iis.seg())->conv->com;
//                        p_first_old = to_xy((*iis.seg())->p.front());
//                        p_last_old  = to_xy((*iis.seg())->p.back());
//                        com_new     = mat_mult(tf.T, com_old);
//                        p_first_new = mat_mult(tf.T, p_first_old);
//                        p_last_new  = mat_mult(tf.T, p_last_old);
//                    }
//                    else{
//                        tf = tf_it->tf_inv;
//                        com_new = (*iis.seg())->conv->com;
//                        p_first_new = to_xy((*iis.seg())->p.front());
//                        p_last_new  = to_xy((*iis.seg())->p.back());
//                        com_old     = mat_mult(tf_it->tf.T, com_new);
//                        p_first_old = mat_mult(tf_it->tf.T, p_first_new);
//                        p_last_old  = mat_mult(tf_it->tf.T, p_last_new);
//                    }

//                    putFullCircle(w2i(tf_sns.s2r(com_new)),1,5,black);

//                    xy arrow_base = w2i(tf_sns.s2r(com_old));
//                    xy arrow_tip  = w2i(tf_sns.s2r(com_old + tf.xy_mean));
//                    putArrow(arrow_base, arrow_tip,green_dark,2);
//                    arrow_base = w2i(tf_sns.s2r(p_first_old));
//                    arrow_tip  = w2i(tf_sns.s2r(p_first_new));
//                    putArrow(arrow_base, arrow_tip,green_dark,2);
//                    arrow_base = w2i(tf_sns.s2r(p_last_old));
//                    arrow_tip  = w2i(tf_sns.s2r(p_last_new));
//                    putArrow(arrow_base, arrow_tip,green_dark,2);


//                    cv::Matx33d cov_xy33(tf.xy_cov(0,0), tf.xy_cov(0,1), 0,
//                                         tf.xy_cov(1,0), tf.xy_cov(1,1), 0,
//                                                     0,               0, 0);
//                    cv::Matx33d Mw2i33(Mw2i);
//                    cov_xy33 = Mw2i33 * cov_xy33 * Mw2i33.t();
//                    cv::Matx22d cov_xy22(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
//                    cv::RotatedRect ellips = cov2rect(cov_xy22,w2i(tf_sns.s2r(com_old + tf.xy_mean)));
//                    cv::ellipse(plot,ellips,green_bright,2);
//                }
//                xy p_tf;
//                if( (*iis.seg())->getObj()){
//                    p_tf = mat_mult(tf_it->tf.T,to_xy(*iis.p()));
//                }
//                else{
//                    p_tf = to_xy(*iis.p());
//                }
//                circle(plot, w2i(tf_sns.s2r(p_tf)), 2, black);
//            }
//        }while( iis.advance(ALL_SEGM, INC));
//    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_kalman(const SegmentDataPtrVectorPtr &data, KalmanSLDM& k, cv::Scalar col_cov_v, cv::Scalar col_cov_x){
    if((!data)||(!k.pos_init)){ return; }

    circle(plot, w2i(0,0), Mw2i[2][2] * 0.15, blue, 2);
    line  (plot, w2i(0,0), w2i(0.15, 0), blue, 2);

    xy pose_corr(k.S.at<double>(0,0) - k.S_bar.at<double>(0,0), k.S.at<double>(1,0) - k.S_bar.at<double>(1,0));
    RState rob_bar(k.S_bar);RState rob_hat(k.S);
    double rob_bar_xphi_f0 = k.S_bar.at<double>(2,0);
    double rob_hat_xphi_f0 = k.S    .at<double>(2,0);
    SensorTf w2r;
    w2r.init(xy(0,0),rob_bar_xphi_f0);
    xy pose_corrected(w2r.r2s(pose_corr));


    if(sqrt(sqr(pose_corr.x) + sqr(pose_corr.y) ) > 0.05){
        putFullCircle(w2i(xy(0,0)),1,3,col_cov_x);
        putArrow(w2i(0,0),w2i(pose_corrected),col_cov_x,2);

        circle(plot, w2i(pose_corrected), Mw2i[2][2] * 0.15, blue_bright, 2);
        double d_ang = k.S.at<double>(2,0) - rob_bar_xphi_f0;
        line  (plot, w2i(pose_corrected), w2i(pose_corrected.x + 0.15 * cos(d_ang), pose_corrected.y + 0.15 * sin(d_ang)), blue_bright, 2);
    }

    cv::Matx33d Mw2i33(Mw2i);
    cv::Matx22d Mw2i22(Mw2i33(0,0),Mw2i33(0,1),Mw2i33(1,0),Mw2i33(1,1));
    cv::Matx33d cov_xy33(k.P.rowRange(0,3).colRange(0,3));

    cv::Matx22d cov_xy22(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));

    cv::Matx22d rot_rob_bar(cos(-rob_bar_xphi_f0),-sin(-rob_bar_xphi_f0),sin(-rob_bar_xphi_f0), cos(-rob_bar_xphi_f0));

    cov_xy22 = Mw2i22 * rot_rob_bar * cov_xy22 *rot_rob_bar.t() * Mw2i22.t();

    cv::RotatedRect ellips = cov2rect(cov_xy22,w2i(pose_corrected));
    cv::ellipse(plot,ellips,col_cov_x,2);

    FrameTf bar2hat;
    bar2hat.init(rob_bar, rob_hat);
    cv::Matx22d rot_rob_hat(cos(-rob_hat_xphi_f0),-sin(-rob_hat_xphi_f0),sin(-rob_hat_xphi_f0), cos(-rob_hat_xphi_f0));
    for(int i=0; i< data->size(); i++){
        xy com  =  bar2hat.rn2ro(data->at(i)->getCom());
        ObjectDataPtr obj = data->at(i)->getObj();
        if( k.Oi.count(obj) != 0 ){
            xy v(k.Oi[obj].S_O.at<double>(3,0),k.Oi[obj].S_O.at<double>(4,0));double w = k.Oi[obj].S_O.at<double>(5,0);
            v = rot_rob_bar * v;

            putFullCircle(w2i(tf_sns.s2r(com)),1,3,col_cov_v);
            if(sqrt(sqr(v.x) + sqr(v.y) ) > k.no_upd_vel_hard0){
                putArrow(w2i(tf_sns.s2r(com)),w2i(tf_sns.s2r(com + v)),col_cov_v,2);
            }

            Mw2i22 = cv::Matx22d(Mw2i33(0,0),Mw2i33(0,1),Mw2i33(1,0),Mw2i33(1,1));
            cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(3,6).colRange(3,6));

            cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));

            cov_xy22 = Mw2i22 * rot_rob_bar * cov_xy22 * rot_rob_bar.t() * Mw2i22.t();

            ellips = cov2rect(cov_xy22,w2i(tf_sns.s2r(com + v)));
            cv::ellipse(plot,ellips,col_cov_v,2);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::plot_corr_links(const std::vector<CorrInput>& list,cv::Scalar color_o2n,cv::Scalar color_n2o){
    for(std::vector<CorrInput>::const_iterator it_corr=list.begin();it_corr!=list.end();it_corr++){
        if( it_corr->reverse ){
            putArrow(w2i(tf_sns.s2r(it_corr->frame_new->getCom())),w2i(tf_sns.s2r(it_corr->frame_old->getCom())),color_n2o,2);
        }
        else{
            putArrow(w2i(tf_sns.s2r(it_corr->frame_old->getCom())),w2i(tf_sns.s2r(it_corr->frame_new->getCom())),color_o2n,2);
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

        for(int i=0; i<50 ; i++){ text[i] = false; }

        plot.release();
        namedWindow(wndView_, 1);
//        cv::moveWindow(wndView_, 0, 0);
        plot = Mat(image_size, image_size + 800, CV_8UC3);
        init_w2i();
        plot(Range(0,image_size),Range(0,image_size)).setTo(white);
        plot(Range(0,image_size),Range(image_size,image_size + 800)).setTo(gray);


        ellipse(plot, w2i(tf_sns.s2r(0,0)), Size( Mw2i[2][2] * sensor_range_max, Mw2i[2][2]* sensor_range_max), rad_to_deg(-acos(Mw2i[0][0])), rad_to_deg(angle_min), rad_to_deg(angle_max), black );
        line(plot, w2i(tf_sns.s2r(0,0)), w2i(tf_sns.s2r(sensor_range_max * cos(angle_max), sensor_range_max * sin(angle_max))), black);
        line(plot, w2i(tf_sns.s2r(0,0)), w2i(tf_sns.s2r(sensor_range_max * cos(angle_min), sensor_range_max * sin(angle_min))), black);


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
        }

    }
    else{
        destroyWindow(wndView_);
        waitKey(10);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotData::putInfoText(std::stringstream &text_data,int row,Scalar color){
    int no_of_lines = 0; int k=0;
    text_data<<std::endl;
    std::stringstream text_data_cpy[2]; text_data_cpy[0].str(text_data.str()); text_data_cpy[1].str(text_data.str());
    std::string line;
    while(std::getline(text_data_cpy[0], line)){
        no_of_lines++;
    }
    while((row < 50)&&(k < no_of_lines)){
        k++;
        if(text[row] == true){
            k=0;
        }
        row++;
    }
    if(row == 50){
        return;
    }
    row = row - no_of_lines;
    while(std::getline(text_data_cpy[1], line)){
        text[row] = true;
        putText(plot,line.c_str(),xy(image_size, 15 * ++row),FONT_HERSHEY_PLAIN,1,color);
    }
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

///------------------------------------------------------------------------------------------------------------------------------------------------///
