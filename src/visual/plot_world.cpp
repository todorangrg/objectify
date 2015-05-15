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
#include "visual/plot_world.h"

#include "boost/type_traits/is_same.hpp"
#include <boost/lexical_cast.hpp>

using namespace cv;


enum{
    NOW,
    FTR
};
enum{
    NEG,
    POS,
    UNK
};



PlotWorld::PlotWorld(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns, KalmanSLDM & _k):
    Plot(wndView),
    plot_world(_param.viz_world),
    plot_grid(_param.viz_world_grid),
    view_len(_param.viz_world_len),
    tf_sns(_tf_sns),
    image_size(900),
    k(_k),
    writing_to_bag(false){

    namedWindow   (wndView_, CV_GUI_EXPANDED);
    resizeWindow  (wndView_, 900, 900);
    cv::moveWindow(wndView_, 900, 0);
    plot = Mat(900, 900, CV_8UC3);
    plot(Range(0,900),Range(0,900)).setTo(white);
    init_w2i();
    // Callback-Function to get location and button of a mouse click in the window
    cvSetMouseCallback ( wndView_.c_str(), this->mouseCallBackWorld, this);


    o_col[0].color = obj_gen_0; o_col[0].used  = false; o_col[0].obj_id = -1;
    o_col[1].color = obj_gen_1; o_col[1].used  = false; o_col[1].obj_id = -1;
    o_col[2].color = obj_gen_2; o_col[2].used  = false; o_col[2].obj_id = -1;
    o_col[3].color = obj_gen_3; o_col[3].used  = false; o_col[3].obj_id = -1;
    o_col[4].color = obj_gen_4; o_col[4].used  = false; o_col[4].obj_id = -1;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::mouseCallBackWorld ( int evt, int c, int r, int flags, void *param ) {
    static int bag_no = 0;
    PlotWorld &pw = *((PlotWorld *) param);
    if      ( evt == CV_EVENT_LBUTTONDOWN ) {
        pw.k.goal = pw.i2w(c,r);
        for(std::map<ObjectDataPtr, ObjMat>::iterator oi =  pw.k.Oi.begin(); oi != pw.k.Oi.end(); oi ++){
            oi->first->dist_to_goal[0] = 10000000;
            oi->first->dist_to_goal[1] = 10000000;
            oi->first->ang_bounds[0][0].r = -1000;
            oi->first->ang_bounds[0][1].r = -1000;
            oi->first->ang_bounds[1][0].r = -1000;
            oi->first->ang_bounds[1][1].r = -1000;
        }//RESET OBJECT T_BUG INFO!!!!!!!!

        pw.k.bag.close();
        pw.writing_to_bag = false;
    }
    else if ( evt == CV_EVENT_RBUTTONDOWN ) {
        if(pw.k.pos_init == false){ return; }
        pw.k.pos_init = false; // resetting kalman

        pw.k.bag.close();
        pw.k.bag.open(pw.k.bag_file_n + boost::lexical_cast<std::string>(bag_no) + ".bag",rosbag::bagmode::Write);
        bag_no++;
        pw.writing_to_bag = true;
    }
    else if ( evt == CV_EVENT_MBUTTONDOWN ) {
        pw.view_center.x =  pw.k.S.at<double>(0);
        pw.view_center.y =  pw.k.S.at<double>(1);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::plot_t_bug(double d_followed, ObjectDataPtr o_followed, int dir_followed, polar target_p, polar target_p_ext, polar potential){
    if(!k.pos_init){ return; }
    std::stringstream s;
    s<<"d_followed= "<<d_followed;
    putText(plot,s.str(), xy(600,35),FONT_HERSHEY_PLAIN,1,green);s.str("");
    s<<"o_followed= ";
    if(d_followed == -1){ s<<"Free space";  }
    else                { s<<o_followed->id; }
    putText(plot,s.str(), xy(600, 65),FONT_HERSHEY_PLAIN,1,green);s.str("");
    s<<"dir_followed= ";
    if     (dir_followed == 0){ s<<"NEG"; }
    else if(dir_followed == 1){ s<<"POS"; }
    else                      { s<<"UNK"; }
    putText(plot,s.str(), xy(600, 95),FONT_HERSHEY_PLAIN,1,green);s.str("");

    RState rob_f0(k.S);
    tf_r.init(xy(rob_f0.xx, rob_f0.xy), rob_f0.xphi);



    line(plot,w2i(tf_r.s2r(tf_sns.s2r(xy(0,0)))),w2i(tf_r.s2r(/*tf_sns.s2r(*/to_xy(target_p)/*)*/)),magenta,2);
    line(plot,w2i(tf_r.s2r(tf_sns.s2r(xy(0,0)))),w2i(tf_r.s2r(/*tf_sns.s2r(*/to_xy(target_p_ext)/*)*/)),red,2);
    line(plot,w2i(tf_r.s2r(tf_sns.s2r(xy(0,0)))),w2i(tf_r.s2r(tf_sns.s2r(to_xy(potential)))),yellow,2);


    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = k.Oi.begin(); oi != k.Oi.end(); oi++){
        cv::Scalar color;
        if(oi->first->ang_b_valid[NOW][POS]){ color = green; } else { color = red; }
        putFullCircle(w2i(tf_r.s2r(/*tf_sns.s2r(*/to_xy(oi->first->ang_bounds[NOW][POS]/*)*/))),1,5,color);
        if(oi->first->ang_b_valid[NOW][NEG]){ color = green; } else { color = red; }
        putFullCircle(w2i(tf_r.s2r(/*tf_sns.s2r(*/to_xy(oi->first->ang_bounds[NOW][NEG]/*)*/))),1,5,color);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::plot_points(PointDataVector &data,Scalar color, SensorTf & tf_r){
    for (PointDataVectorIter p = data.begin(); p != data.end(); p++){
        putFullCircle(w2i(tf_r.s2r(tf_sns.s2r(to_xy(*p)))),3,6,color);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void PlotWorld::plot_segm(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data,cv::Scalar color, SensorTf & tf_r){
    if(!data){
        return;
    }
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator it_data=data->begin(); it_data != data->end(); it_data++){

        plot_points((*it_data)->p, color, tf_r);

        xy com = (*it_data)->getCom();
        putFullCircle(w2i(tf_sns.s2r(com)),1,4,color);

        std::stringstream s; s.precision(4);
        s//<<"l:" <<std::setw(3)<< (*it_data)->getLen()
         <<"|S:"<<               (*it_data)->id
         <<"|O:";
        bool frame_old, init;
        if((*it_data)->getObj()){                        s << (*it_data)->getObj()->id << "(t-1)"; frame_old = true; }
        else{                                            s << "?"                      << "( t )"; frame_old = false;}
        if(boost::is_same<SegData, SegmentData>::value){ s << "i";                                 init      = true; }
        else{                                            s << "e";                                 init      = false;}
        xy direct;
        double len = 0.5;
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
        line   (plot, w2i(tf_sns.s2r(com)), w2i(tf_sns.s2r(com + direct)), color);
        putText(plot,s.str().c_str(),w2i(tf_sns.s2r(com + direct)),FONT_HERSHEY_PLAIN,1,color);
    }
}

template void PlotWorld::plot_segm<SegmentData>   (SegmentDataPtrVectorPtr    &data, cv::Scalar color, SensorTf & tf_r);
template void PlotWorld::plot_segm<SegmentDataExt>(SegmentDataExtPtrVectorPtr &data, cv::Scalar color, SensorTf & tf_r);

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void PlotWorld::plot_kalman(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > _input, cv::Scalar col_cov_x, cv::Scalar col_cov_v){
    if(!k.pos_init){ return; }
    cv::Matx33d Mw2i33(Mw2i);
    cv::Matx22d Mw2i22(Mw2i33(0,0),Mw2i33(0,1),Mw2i33(1,0),Mw2i33(1,1));
    cv::Matx33d cov_xy33;
    cv::Matx22d cov_xy22;
    cv::RotatedRect ellipse;

    cov_xy33 = cv::Matx33d (k.P.rowRange(0,3).colRange(0,3));
    cov_xy22 = cv::Matx22d (cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
    cov_xy22 = Mw2i22 * cov_xy22 * Mw2i22.t();

    RState rob_f0(k.S);
    tf_r.init(xy(rob_f0.xx, rob_f0.xy), rob_f0.xphi);

    putFullCircle(w2i(rob_f0.xx, rob_f0.xy), Mw2i[2][2] * 0.15 - 2, Mw2i[2][2] * 0.15 + 5,blue);
    line  (plot, w2i(rob_f0.xx, rob_f0.xy), w2i(rob_f0.xx + 0.15 * cos(rob_f0.xphi), rob_f0.xy + 0.15 * sin(rob_f0.xphi)), blue, 3);

    ellipse = cov2rect(cov_xy22,w2i(rob_f0.xx, rob_f0.xy));
    cv::ellipse(plot,ellipse,col_cov_x,3);

    putFullCircle(w2i(rob_f0.xx, rob_f0.xy),1,3,col_cov_v);
    if(fabs(k.S.at<double>(3)) > 0.2){
        putArrow(w2i(rob_f0.xx, rob_f0.xy),w2i(rob_f0.xx + k.S.at<double>(3) * cos(rob_f0.xphi), rob_f0.xy + k.S.at<double>(3) * sin(rob_f0.xphi)),col_cov_v,3);
    }

    cov_xy22 = cv::Matx22d(k.P.rowRange(3,5).colRange(3,5));
    cv::Matx22d rot_rob(cos(-rob_f0.xphi), sin(-rob_f0.xphi), -sin(-rob_f0.xphi), cos(-rob_f0.xphi));
    cov_xy22 = Mw2i22 * rot_rob * cov_xy22 * rot_rob.t() * Mw2i22.t();
    ellipse = cov2rect(cov_xy22,w2i(rob_f0.xx + k.S.at<double>(3) * cos(rob_f0.xphi), rob_f0.xy + k.S.at<double>(3) * sin(rob_f0.xphi)));
    cv::ellipse(plot,ellipse,col_cov_v,3);

    for(int i = 0; i < 5; i++){
        o_col[i].used = false;
    }


    for(typename std::vector<boost::shared_ptr<SegData> >::iterator s_it = _input->begin(); s_it != _input->end(); s_it++){

        xy com  =  tf_r.s2r(tf_sns.s2r((*s_it)->getCom()));
        ObjectDataPtr obj = (*s_it)->getObj();

        cv::Scalar color_obj;
        bool found_col = false;
        for(int i = 0; i < 5; i++){
            if(o_col[i].obj_id == obj->id){
                color_obj = o_col[i].color;
                o_col[i].used = true;
                found_col = true;
                break;
            }
        }
        if(!found_col){
            for(int i = 0; i < 5; i++){
                if(o_col[i].obj_id == -1){
                    color_obj = o_col[i].color;
                    o_col[i].used = true;
                    o_col[i].obj_id = obj->id;
                    break;
                }
            }
        }


        double inc = fmin(1.0, obj->life_time / 2.0);
        plot_points((*s_it)->p, cv::Scalar(0,0,0) + inc * color_obj, tf_r);

        if( k.Oi.count(obj) != 0 ){
            //if plot vel
            xy     v  (k.Oi[obj].S_O.at<double>(3,0),k.Oi[obj].S_O.at<double>(4,0));
            double w = k.Oi[obj].S_O.at<double>(5,0);
//            std::cout<<"w_state = "<<w<<std::endl;

            putFullCircle(w2i(com),1,3,col_cov_v);
            if(sqrt(sqr(v.x) + sqr(v.y) ) > 0.2){
                putArrow(w2i(com),w2i(com + v),col_cov_v,3);
            }

            cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(3,6).colRange(3,6));
            cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
            cov_xy22 = Mw2i22 * cov_xy22 * Mw2i22.t();
            ellipse = cov2rect(cov_xy22,w2i(com + v));
            cv::ellipse(plot,ellipse,col_cov_v,3);

            //if plot pos_cov

            cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(0,3).colRange(0,3));
            cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
            cov_xy22 = Mw2i22 * cov_xy22 * Mw2i22.t();

            cv::RotatedRect ellipse = cov2rect(cov_xy22,w2i(com));
            cv::ellipse(plot,ellipse,col_cov_x,3);
        }
    }

    for(int i = 0; i < 5; i++){
        if(o_col[i].used == false){
            o_col[i].obj_id = -1;
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::update(){
    if(plot_world){
        imshow(wndView_,plot);
        waitKey(10);

        plot.release();
        namedWindow(wndView_, 1);
//        cv::moveWindow(wndView_, 0, 0);
        plot = Mat(image_size, image_size, CV_8UC3);
        init_w2i();
        plot(Range(0,image_size),Range(0,image_size)).setTo(white);

        if (plot_grid) {
            for (double y = (int)(view_center.y - view_len / 2.0) + 1; y <= (int)(view_center.y + view_len / 2.0); y+=1.0) {
                for (double x = (int)(view_center.x - view_len / 2.0) + 1; x <= (int)(view_center.x + view_len / 2.0); x+=1.0) {
                    circle(plot, w2i(x,y), 1, black);
                }
            }
            line(plot, w2i(-5.0,0), w2i(5.0,0), black);
            line(plot, w2i(0,-5.0) , w2i(0,5.0), black);
            putText(plot, "X [5m]", w2i(5.2, -0.3), FONT_HERSHEY_PLAIN, 1, black);
            putText(plot, "Y [5m]", w2i(0.2, 5.5) , FONT_HERSHEY_PLAIN, 1, black);
        }
        else{
            double scale  = view_len / 8.0 + 0.05;
            int i=0;
//            for (double y = (view_center.y - view_len / 2.0); y < (view_center.y + view_len / 2.0); y+=scale) {
//                if(i==0){i++;continue;}
//                std::stringstream s; s.precision(2);s<<std::setw(3)<<std::fixed<<y;
//                putText(plot, s.str().c_str(), cv::Point(-16,-15) + w2i((view_center.x - view_len / 2.0),y), FONT_HERSHEY_PLAIN, 1, black);
//                putArrow(w2i((view_center.x - view_len / 2.0),y), cv::Point(0,-10) + w2i((view_center.x - view_len / 2.0),y), black);
//                i++;
//            }
//            i=0;
//            for (double x = (view_center.x + view_len / 2.0); x > (view_center.x - view_len / 2.0); x-=scale) {
//                if(i==0){i++;continue;}
//                std::stringstream s; s.precision(2);s<<std::setw(3)<<std::fixed<<x;
//                putText(plot, s.str().c_str(), cv::Point(15,6) + w2i(x,(view_center.y + view_len / 2.0)), FONT_HERSHEY_PLAIN, 1, black);
//                putArrow(w2i(x,(view_center.y + view_len / 2.0)), cv::Point(10,0) + w2i(x,(view_center.y + view_len / 2.0)), black);

//            }
//            putText(plot, "[X]",cv::Point(5,20) + w2i((view_center.x + view_len / 2.0), (view_center.y + view_len / 2.0)), FONT_HERSHEY_PLAIN, 1, black);
//            putText(plot, "[Y]",cv::Point(-50,-5) + w2i((view_center.x - view_len / 2.0), (view_center.y - view_len / 2.0)), FONT_HERSHEY_PLAIN, 1, black);

//            line(plot, w2i(-5.0,0), w2i(5.0,0), black);
//            line(plot, w2i(0,-5.0) , w2i(0,5.0), black);
//            putText(plot, "X [5m]", w2i(5.2, -0.3), FONT_HERSHEY_PLAIN, 1, black);
//            putText(plot, "Y [5m]", w2i(0.2, 5.5) , FONT_HERSHEY_PLAIN, 1, black);
        }


//        putFullCircle(w2i(k.goal),1,5,blue);

        plot_kalman(k.seg_init, cov_x, cov_v);
        if(writing_to_bag){
            //putFullCircle(xy(30,30),1,8,red);
            //putText(plot,"Recording to bag-file", xy(42,38),FONT_HERSHEY_PLAIN,1,red);
        }
    }
    else{
        destroyWindow(wndView_);
        waitKey(10);
    }
}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::init_w2i() {
    double scale = (double)image_size / ( view_len );
    Mat_<double> Sc = (Mat_<double>(3,3) << scale, 0,     0,
                                            0, scale,     0,
                                            0,     0, scale);   // Scale
    Mat_<double> M = (Mat_<double>(3,3) << -1, 0, 0,
                                           0, 1, 0,
                                           0, 0, 1);    // Mirror
    Mat_<double> R = (Mat_<double>(3,3) << 0, -1, 0,
                                           1, 0, 0,
                                           0, 0, 1);    // Rotate
    Mat_<double> T = (Mat_<double>(3,3) << 1, 0, view_len / 2.0 + view_center.y,
                                           0, 1, view_len / 2.0 + view_center.x,
                                           0, 0, 1);    // Translate

    // Calculate final transformation matrix
    Mw2i = T * R * M * Sc;
    Mi2w = Mw2i.inv(cv::DECOMP_SVD);
//    std::cout <<  "Mw2i = " << std::endl << Mw2i << std::endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
