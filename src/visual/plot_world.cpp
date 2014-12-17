
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

PlotWorld::PlotWorld(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns, KalmanSLDM & _k):
    Plot(wndView),
    plot_world(_param.viz_world),
    plot_grid(_param.viz_world_grid),
    view_len(_param.viz_world_len),
    tf_sns(_tf_sns),
    image_size(900),
    k(_k){

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
        if(pw.k.pos_init == false){ return; }
        pw.view_center.x =  pw.k.S.at<double>(0);
        pw.view_center.y =  pw.k.S.at<double>(1);
        pw.k.pos_init = false; // resetting kalman

        pw.k.bag.close();
        pw.k.bag.open(pw.k.bag_file_n + boost::lexical_cast<std::string>(bag_no) + ".bag",rosbag::bagmode::Write);
        bag_no++;
    }
    else if ( evt == CV_EVENT_RBUTTONDOWN ) {
    }
    else if ( evt == CV_EVENT_MBUTTONDOWN ) {
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::plot_points(PointDataVector &data,Scalar color){
    for (PointDataVectorIter p = data.begin(); p != data.end(); p++){
        circle(plot, w2i(tf_r.s2r(tf_sns.s2r(to_xy(*p)))),3,color);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void PlotWorld::plot_segm(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data,cv::Scalar color){
    if(!data){
        return;
    }
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator it_data=data->begin(); it_data != data->end(); it_data++){

        plot_points((*it_data)->p, color);

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

template void PlotWorld::plot_segm<SegmentData>   (SegmentDataPtrVectorPtr    &data, cv::Scalar color);
template void PlotWorld::plot_segm<SegmentDataExt>(SegmentDataExtPtrVectorPtr &data, cv::Scalar color);

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

    circle(plot, w2i(rob_f0.xx, rob_f0.xy), Mw2i[2][2] * 0.3, blue, 3);
    line  (plot, w2i(rob_f0.xx, rob_f0.xy), w2i(rob_f0.xx + 0.3 * cos(rob_f0.xphi), rob_f0.xy + + 0.3 * sin(rob_f0.xphi)), blue, 3);

    ellipse = cov2rect(cov_xy22,w2i(rob_f0.xx, rob_f0.xy));
    cv::ellipse(plot,ellipse,col_cov_x,2);

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
        plot_points((*s_it)->p, cv::Scalar(0,0,0) + inc * color_obj);

        if( k.Oi.count(obj) != 0 ){
            //if plot vel
            xy     v  (k.Oi[obj].S_O.at<double>(3,0),k.Oi[obj].S_O.at<double>(4,0));
            double w = k.Oi[obj].S_O.at<double>(5,0);

            putFullCircle(w2i(com),1,3,col_cov_v);
            if(sqrt(sqr(v.x) + sqr(v.y) ) > 0.2){
                putArrow(w2i(com),w2i(com + v),col_cov_v,2);
            }

            cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(3,6).colRange(3,6));
            cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
            cov_xy22 = Mw2i22 * cov_xy22 * Mw2i22.t();
            ellipse = cov2rect(cov_xy22,w2i(com + v));
            cv::ellipse(plot,ellipse,col_cov_v,2);

            //if plot pos_cov

            cov_xy33 = cv::Matx33d(k.Oi[obj].P_OO.rowRange(0,3).colRange(0,3));
            cov_xy22 = cv::Matx22d(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));
            cov_xy22 = Mw2i22 * cov_xy22 * Mw2i22.t();

            cv::RotatedRect ellipse = cov2rect(cov_xy22,w2i(com));
            cv::ellipse(plot,ellipse,col_cov_x,2);
        }
    }

    for(int i = 0; i < 5; i++){
        if(o_col[i].used == false){
            o_col[i].obj_id = -1;
        }
    }
    int xxx=0;
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


        plot_kalman(k.seg_init, cov_x, cov_v);
    }
    else{
        destroyWindow(wndView_);
        waitKey(10);
    }
}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotWorld::init_w2i() {
    int scale = (double)image_size / ( view_len );
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
    Mi2w = Mw2i.inv();
//    std::cout <<  "Mw2i = " << std::endl << Mw2i << std::endl;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///