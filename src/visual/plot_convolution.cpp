
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "utils/iterators.h"
#include "utils/base_classes.h"
#include "data_processing/dp.h"

#include "utils/convolution.h"
#include "visual/plot_convolution.h"


using namespace cv;


PlotConv::PlotConv(std::string wndView,RecfgParam &_param):
    Plot(wndView),
    plot_conv(_param.viz_convol),
    plot_normals(_param.viz_convol_normals),
    plot_tf(_param.viz_convol_tf),
    image_size(500),
    com_dr_thres(_param.convol_com_dr_thres),
    ang_var_thres(_param.convol_ang_var_thres),
    sqr_err_thres(_param.convol_sqr_err_thres),
    score_thres(_param.convol_score_thres),
    convol_no(_param.viz_convol_step_no),
    tf_ref2spl(_param.viz_convol_tf_ref2spl){

    x_border[0] = 0.02*image_size;
    x_border[1] = 0.02*image_size;
    y_border[0] = 30;
    y_border[1] = 30;

    for(int i=0;i<no_graph_param;i++){
        gdata[i].max_thrs = NULL;
        gdata[i].min_thrs = NULL;
    }

    gdata[ANG_VAR].name<<"ang_var";
    gdata[ANG_VAR].max_thrs = &ang_var_thres;
    gdata[ANG_VAR].color = red;

    gdata[DR_MEAN].name<<"dr_mean";
    gdata[DR_MEAN].max_thrs = &com_dr_thres;
    gdata[DR_MEAN].color = cyan;

    gdata[P_PAIR].name<<"p_pair";
    gdata[P_PAIR].color = green;

    gdata[SQR_ERR].name<<"sqr_err";
    gdata[SQR_ERR].max_thrs = &sqr_err_thres;
    gdata[SQR_ERR].color = magenta;

    gdata[SCORE].name<<"score";
//    gdata[SCORE].min_thrs = &score_thres;
    gdata[SCORE].color = blue;

    font_size = 0.8;

    y_border[0] = 3*10*font_size;
    y_border[1] = 3*10*font_size;

    namedWindow(wndView_,CV_GUI_EXPANDED);
    resizeWindow(wndView_,image_size*2,image_size);
    cv::moveWindow(wndView_, 0, 600);
    plot = Mat(image_size, image_size*2, CV_8UC3);
    plot(Range(0,image_size),Range(0,image_size)).setTo(white);
    plot(Range(0,image_size),Range(image_size,image_size*2)).setTo(gray);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::plot_grids(){
    std::stringstream s;
    s.precision(2);

    line(plot,cv::Point2i(image_size,font_size*10*3),cv::Point2i(image_size*2,font_size*10*3), black);
    line(plot,cv::Point2i(image_size,image_size -font_size*10*3),cv::Point2i(image_size*2,image_size -font_size*10*3), black);

    for(int i=0;i<no_graph_param;i++){
        double max_thrs,min_thrs;
        if(gdata[i].max_thrs == NULL){
            max_thrs = gdata[i].max;
        }
        else{
            max_thrs = *gdata[i].max_thrs;
        }
        if(gdata[i].min_thrs == NULL){
            min_thrs = gdata[i].min;
        }
        else{
            min_thrs = *gdata[i].min_thrs;
        }


        putText(plot, gdata[i].name.str().c_str(), cv::Point2i(image_size-70,image_size-font_size*10*(i+1) ), FONT_HERSHEY_PLAIN, font_size, gdata[i].color);

        s<<std::setw(4)<<std::fixed<<gdata[i].max;
        putText(plot, s.str().c_str(), cv::Point2i(image_size+i*image_size/2.0/(double)no_graph_param,font_size*10*2),FONT_HERSHEY_PLAIN,font_size,gdata[i].color); s.str(std::string());
        s<<std::setw(4)<<std::fixed<<gdata[i].min;
        putText(plot, s.str().c_str(), cv::Point2i(image_size+i*image_size/2.0/(double)no_graph_param,image_size-font_size*10*(2-1)),FONT_HERSHEY_PLAIN,font_size,gdata[i].color); s.str(std::string());


        s<<std::setw(4)<<std::fixed<<max_thrs;
        putText(plot, s.str().c_str(), cv::Point2i(image_size+image_size/2.0+i*image_size/2.0/(double)no_graph_param,font_size*10*2),FONT_HERSHEY_PLAIN,font_size,gdata[i].color); s.str(std::string());
        s<<std::setw(4)<<std::fixed<<min_thrs;
        putText(plot, s.str().c_str(), cv::Point2i(image_size+image_size/2.0+i*image_size/2.0/(double)no_graph_param,image_size-font_size*10*(2-1)),FONT_HERSHEY_PLAIN,font_size,gdata[i].color); s.str(std::string());

        int y_pos;
        y_pos = image_size - y_border[0] - fmin(gdata[i].scale*(max_thrs- gdata[i].min),image_size - 2* y_border[0]);
        line(plot,cv::Point2i(image_size,y_pos),cv::Point2i(image_size * 2,y_pos),gdata[i].color);
        putText(plot,"v", cv::Point2i(image_size         ,y_pos+10*font_size),FONT_HERSHEY_PLAIN,font_size,gdata[i].color);
        putText(plot,"v", cv::Point2i(image_size * 2 - 10,y_pos+10*font_size),FONT_HERSHEY_PLAIN,font_size,gdata[i].color);
        y_pos = image_size  - y_border[0] - fmax(gdata[i].scale* (min_thrs- gdata[i].min),0);
        line(plot,cv::Point2i(image_size,y_pos),cv::Point2i(image_size * 2,y_pos),gdata[i].color);
        putText(plot,"^", cv::Point2i(image_size         ,y_pos),FONT_HERSHEY_PLAIN,font_size,gdata[i].color);
        putText(plot,"^", cv::Point2i(image_size * 2 - 10,y_pos),FONT_HERSHEY_PLAIN,font_size,gdata[i].color);


    }
    for(int i=0;i<no_graph_param;i++){
        double max_thrs,min_thrs;
        if(gdata[i].max_thrs == NULL){
            max_thrs = gdata[i].max;
        }
        else{
            max_thrs = *gdata[i].max_thrs;
        }
        if(gdata[i].min_thrs == NULL){
            min_thrs = gdata[i].min;
        }
        else{
            min_thrs = *gdata[i].min_thrs;
        }

        int no_seg_lines = 5;
        int y_pos;
        for(int j=0;j<no_seg_lines*no_graph_param;j=j+no_graph_param){
            y_pos = image_size - y_border[0] - fmin(gdata[i].scale*(max_thrs- gdata[i].min),image_size - 2* y_border[0]);
            line(plot,cv::Point2i(image_size+(i+j)*image_size/(no_seg_lines*no_graph_param),y_pos),cv::Point2i(image_size+(i+j+1)*image_size/(no_seg_lines*no_graph_param),y_pos),gdata[i].color);
            y_pos = image_size  - y_border[0] - fmax(gdata[i].scale* (min_thrs- gdata[i].min),0);
            line(plot,cv::Point2i(image_size+(i+j)*image_size/(no_seg_lines*no_graph_param),y_pos),cv::Point2i(image_size+(i+j+1)*image_size/(no_seg_lines*no_graph_param),y_pos),gdata[i].color);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::load_graph_param(std::vector<boost::shared_ptr<ConvolInfo> >& convol_distr){
    for(int i=0;i<no_graph_param;i++){
        gdata[i].max = 0;
        gdata[i].min = 0;
    }
    for(int i=0;i<convol_distr.size();i++){
        if( convol_distr[i]->ang_distr.getVariance()>gdata[ANG_VAR].max ){
            gdata[ANG_VAR].max = convol_distr[i]->ang_distr.getVariance();
        }
        if( convol_distr[i]->com_dr>gdata[DR_MEAN].max ){
            gdata[DR_MEAN].max = convol_distr[i]->com_dr;
        }
        if( convol_distr[i]->pair_no>gdata[P_PAIR].max ){
            gdata[P_PAIR].max = convol_distr[i]->pair_no;
        }
        if( convol_distr[i]->sqr_err>gdata[SQR_ERR].max ){
            gdata[SQR_ERR].max = convol_distr[i]->sqr_err;
        }
        if( convol_distr[i]->score<gdata[SCORE  ].min ){
            gdata[SCORE  ].min = convol_distr[i]->score;
        }
        if( convol_distr[i]->score>gdata[SCORE  ].max ){
            gdata[SCORE  ].max = convol_distr[i]->score;
        }
    }
    scale_x      = (image_size-(x_border[0]+x_border[1])) / convol_distr.size();
    for(int i=0;i<no_graph_param;i++){
       gdata[i].scale = (image_size-(y_border[0]+y_border[1])) / (gdata[i].max-gdata[i].min);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::plot_conv_info(std::vector<boost::shared_ptr<ConvolInfo> >& convol_distr,cv::Scalar color_score,cv::Scalar color_var,cv::Scalar color_dist){
    if( convol_distr.size() == 0 ){
        return;
    }
    cv::Scalar color_score0 = color_score;

    load_graph_param(convol_distr);
    plot_grids();
    if( convol_no != 0 ){
        line(plot,cv::Point2i(image_size + x_border[0]+ scale_x*(min(convol_no,(int)convol_distr.size())-1),y_border[0]),cv::Point2i(image_size + x_border[0]+ scale_x*(min(convol_no,(int)convol_distr.size())-1),image_size-y_border[0]),black);
    }


    if(convol_distr[0]->score> 0.0 ){
        color_score = yellow;
    }


    circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*( 0 ),image_size - y_border[0] - gdata[ANG_VAR ].scale * (convol_distr[0]->ang_distr.getVariance() - gdata[ANG_VAR ].min)), 2, gdata[ANG_VAR].color);
    circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*( 0 ),image_size - y_border[0] - gdata[DR_MEAN ].scale * (convol_distr[0]->com_dr                  - gdata[DR_MEAN ].min)), 2, gdata[DR_MEAN].color);
    circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*( 0 ),image_size - y_border[0] - gdata[P_PAIR  ].scale * (convol_distr[0]->pair_no                 - gdata[P_PAIR  ].min)), 2, gdata[P_PAIR ].color);
    circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*( 0 ),image_size - y_border[0] - gdata[SQR_ERR ].scale * (convol_distr[0]->sqr_err                 - gdata[SQR_ERR ].min)), 2, gdata[SQR_ERR].color);
    circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*( 0 ),image_size - y_border[0] - gdata[SCORE   ].scale * (convol_distr[0]->score                   - gdata[SCORE   ].min)), 2, color_score);
    for(int i=1;i<convol_distr.size();i++){
        color_score = color_score0;
        line(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i-1),image_size - y_border[0] - gdata[ANG_VAR ].scale * (convol_distr[i-1]->ang_distr.getVariance() - gdata[ANG_VAR ].min)),
                   cv::Point2i(image_size + x_border[0]+ scale_x*( i ),image_size - y_border[0] - gdata[ANG_VAR ].scale * (convol_distr[ i ]->ang_distr.getVariance() - gdata[ANG_VAR ].min)), gdata[ANG_VAR].color);

        line(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i-1),image_size - y_border[0] - gdata[DR_MEAN ].scale * (convol_distr[i-1]->com_dr                  - gdata[DR_MEAN ].min)),
                   cv::Point2i(image_size + x_border[0]+ scale_x*( i ),image_size - y_border[0] - gdata[DR_MEAN ].scale * (convol_distr[ i ]->com_dr                  - gdata[DR_MEAN ].min)), gdata[DR_MEAN].color);

        line(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i-1),image_size - y_border[0] - gdata[P_PAIR  ].scale * (convol_distr[i-1]->pair_no                 - gdata[P_PAIR  ].min)),
                   cv::Point2i(image_size + x_border[0]+ scale_x*( i ),image_size - y_border[0] - gdata[P_PAIR  ].scale * (convol_distr[ i ]->pair_no                 - gdata[P_PAIR  ].min)), gdata[P_PAIR ].color);

        line(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i-1),image_size - y_border[0] - gdata[SQR_ERR ].scale * (convol_distr[i-1]->sqr_err                 - gdata[SQR_ERR ].min)),
                   cv::Point2i(image_size + x_border[0]+ scale_x*( i ),image_size - y_border[0] - gdata[SQR_ERR ].scale * (convol_distr[ i ]->sqr_err                 - gdata[SQR_ERR ].min)), gdata[SQR_ERR].color);

        line(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i-1),image_size - y_border[0] - gdata[SCORE   ].scale * (convol_distr[i-1]->score                   - gdata[SCORE   ].min)),
                   cv::Point2i(image_size + x_border[0]+ scale_x*( i ),image_size - y_border[0] - gdata[SCORE   ].scale * (convol_distr[ i ]->score                   - gdata[SCORE   ].min)), color_score);

        if(convol_distr[i]->score> 0.0 ){
            color_score = yellow;
        }        
        circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i),image_size - y_border[0] - gdata[ANG_VAR ].scale * (convol_distr[i]->ang_distr.getVariance() - gdata[ANG_VAR ].min)), 2, gdata[ANG_VAR].color);
        circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i),image_size - y_border[0] - gdata[DR_MEAN ].scale * (convol_distr[i]->com_dr                  - gdata[DR_MEAN ].min)), 2, gdata[DR_MEAN].color);
        circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i),image_size - y_border[0] - gdata[P_PAIR  ].scale * (convol_distr[i]->pair_no                 - gdata[P_PAIR  ].min)), 2, gdata[P_PAIR ].color);
        circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i),image_size - y_border[0] - gdata[SQR_ERR ].scale * (convol_distr[i]->sqr_err                 - gdata[SQR_ERR ].min)), 2, gdata[SQR_ERR].color);
        circle(plot, cv::Point2i(image_size + x_border[0]+ scale_x*(i),image_size - y_border[0] - gdata[SCORE   ].scale * (convol_distr[i]->score                   - gdata[SCORE   ].min)), 2, color_score);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::update(){
    if(plot_conv){
        imshow(wndView_,plot);
        waitKey(10);

        plot.release();
        namedWindow(wndView_,CV_GUI_EXPANDED);
        resizeWindow(wndView_,image_size*2,image_size);
//        cv::moveWindow(wndView_, 0, 600);
        plot = Mat(image_size, image_size*2, CV_8UC3);
        plot(Range(0,image_size),Range(0,image_size)).setTo(white);
        plot(Range(0,image_size),Range(image_size,image_size*2)).setTo(gray);
        line(plot,cv::Point2i(image_size,image_size),cv::Point2i(image_size,0), black);

    }
    else{
        destroyWindow(wndView_);
        waitKey(10);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::plot_conv_points(std::vector<boost::shared_ptr<ConvolInfo> > &convol_distr, boost::shared_ptr<ConvData> *conv_data){
    init_w2i(conv_data);//scale and center the 2 snapp-able images

    //which tf to plot
    cv::Matx33d T(0,0,0,0,0,0,0,0,0); xy comref,comspl; bool T_valid = false; ConvolStatus conv_stat;
    if(tf_ref2spl){conv_stat = CONV_REF;}
    else{  conv_stat = CONV_SPL;    }
    int tf_arrow_no = 20;
    int points_size = conv_data[conv_stat]->p_cd->size();
    if( points_size < tf_arrow_no ){ tf_arrow_no = 1; }
    else                           { tf_arrow_no = points_size / tf_arrow_no; }

    if( convol_no != 0 ){
        int index = std::min(convol_no-1,(int)convol_distr.size()-1);
        T = convol_distr[index]->T;
        if(conv_stat == CONV_SPL){ T = T.inv(); }
        comref = convol_distr[index]->com_ref; comspl = convol_distr[index]->com_spl;
    }
    else{
        comref = conv_data[CONV_REF]->com; comspl = conv_data[CONV_SPL]->com;
    }

    putFullCircle(w2i(comref),1,5,blue); putFullCircle(w2i(comspl),1,5,red);

    for(int i=0;i<9;i++){ if( T(i) != 0 ){ T_valid = true; break; } }

    for(int i_dir = 0; i_dir < 2; i_dir++){

        for( int i=0;i<conv_data[i_dir]->p_cd->size();i++ ){
            xy p = to_xy(conv_data[i_dir]->p_cd->at(i));
            Scalar color;
            if(i_dir == CONV_SPL){ color = Scalar(0xFF * conv_data[i_dir]->p_cd->at(i).fade_out, 0xFF * conv_data[i_dir]->p_cd->at(i).fade_out, 0xFF); }
            else                 { color = Scalar(0xFF, 0xFF * conv_data[i_dir]->p_cd->at(i).fade_out, 0xFF * conv_data[i_dir]->p_cd->at(i).fade_out); }
            circle(plot,w2i(p),2,color);//plot the points

            if(plot_normals){//plot the normals
                xy arrow_base = p;
                xy arrow_tip  = xy(p.x + 0.1 * Mw2i_scale * cos(conv_data[i_dir]->p_cd->at(i).normal_ang), p.y + 0.1 * Mw2i_scale * sin(conv_data[i_dir]->p_cd->at(i).normal_ang) );
                if(i_dir == CONV_SPL){ color = orange; }
                else                 { color = blue_bright; }
                putArrow(w2i(arrow_base), w2i(arrow_tip ), color, 1);
            }
            if((plot_tf)&&(conv_stat == i_dir)){//plot the tf
                if( convol_no != 0 ){
                    if(T_valid){//if tf is null, dont plot
                        circle(plot, w2i(mat_mult(T,p)), 2, black);
                        if(i % tf_arrow_no == 0){
                            putArrow(w2i(p), w2i(mat_mult(T,p)), green);
                        }
                    }
                }
                else{
                    for(int j=0; j < conv_data[conv_stat]->tf->size(); j++){
                        xy p_tf = mat_mult(conv_data[conv_stat]->tf->at(j).tf.T,p);
                        circle(plot, w2i(p_tf), 2, black);
                    }
                }
            }
        }
    }
    cv::Matx33d Mw2i33(Mw2i);
    if( convol_no == 0 ){
        std::stringstream s;
        Scalar color;
        if(conv_stat == CONV_SPL){ color = red;}
        else                     { color = blue; }
        s.precision(4);
        s <<"  S:"<<conv_data[conv_stat]->seg->id<<"|O:";
        if(conv_data[conv_stat]->seg->getObj()){
            s<<conv_data[conv_stat]->seg->getObj()->id;
        }
        if(conv_data[conv_stat]->seg->getObj()){s<<" t-1";}
        else                                   {s<<"  t ";}
        putText(plot,s.str().c_str(),w2i(conv_data[conv_stat]->com),FONT_HERSHEY_PLAIN,1,color);s.str(std::string());

        for(int j=0; j < conv_data[conv_stat]->tf->size(); j++){
            putFullCircle(w2i(mat_mult(conv_data[conv_stat]->tf->at(j).tf.T, conv_data[conv_stat]->com)),1,5,black);

            if(conv_stat == CONV_SPL){ color = blue;}
            else                     { color = red; }

            s.precision(4);
            s <<"  S:"<<conv_data[conv_stat]->tf->at(j).seg->id<<"|O:";
            if(conv_data[conv_stat]->tf->at(j).seg->getObj()){
                s<<conv_data[conv_stat]->tf->at(j).seg->getObj()->id;
            }
            if(conv_data[conv_stat]->tf->at(j).seg->getObj()){s<<" t-1";}
            else                                             {s<<"  t ";}
            s<<"w= "<<conv_data[conv_stat]->tf->at(j).score;

            xy    com = conv_data[conv_stat]->com;
            xy tf_com = mat_mult(conv_data[conv_stat]->tf->at(j).tf.T, com);

            putText(plot,s.str().c_str(),w2i(tf_com),FONT_HERSHEY_PLAIN,1,color);s.str(std::string());
            cv::Matx33d cov_xy33(conv_data[conv_stat]->tf->at(j).tf.Q);
            cov_xy33(2,2) = 0.;

            cov_xy33 = Mw2i33 * cov_xy33 * Mw2i33.t();

            cv::Matx22d cov_xy22(cov_xy33(0,0),cov_xy33(0,1),cov_xy33(1,0),cov_xy33(1,1));

            putArrow(w2i(com), w2i(tf_com),green_dark,2);
            cv::RotatedRect ellips = cov2rect(cov_xy22,w2i(tf_com));
            cv::ellipse(plot,ellips,green_dark,2);
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void PlotConv::init_w2i(boost::shared_ptr<ConvData>* conv_data) {
    xy p_min(1000,1000);
    xy p_max(-1000,-1000);
    for(int at=0;at<2;at++  ){
        for( int i=0;i<conv_data[at]->p_cd->size();i++ ){
            xy p = to_xy(conv_data[at]->p_cd->at(i));
            if(p.x<p_min.x){p_min.x = p.x;}
            if(p.y<p_min.y){p_min.y = p.y;}
            if(p.x>p_max.x){p_max.x = p.x;}
            if(p.y>p_max.y){p_max.y = p.y;}
        }
    }
    double scale = p_max.y-p_min.y;
    if( p_max.y-p_min.y < p_max.x-p_min.x ){
        scale = p_max.x-p_min.x;
    }
    double sscale = 0.8 * image_size / scale;
    Mw2i_scale = scale;

    xy shift;
    shift.x = (p_min.x+p_max.x)/2.0;
    shift.y = (p_min.y+p_max.y)/2.0;
    Mat_<double> Sc = (Mat_<double>(3,3) << sscale, 0, 0,
                                            0, sscale, 0,
                                            0, 0, sscale);   // Scale
    Mat_<double> M = (Mat_<double>(3,3) << -1, 0, 0,
                                           0, 1, 0,
                                           0, 0, 1);    // Mirror
    Mat_<double> R = (Mat_<double>(3,3) << 0, -1, 0,
                                           1, 0, 0,
                                           0, 0, 1);    // Rotate
    Mat_<double> T = (Mat_<double>(3,3) << 1, 0, -shift.x,
                                           0, 1, -shift.y,
                                           0, 0, 1);    // Translate
    Mat_<double> Tf = (Mat_<double>(3,3) << 1, 0, image_size/(2.0*sscale),
                                            0, 1, image_size/(2.0*sscale),
                                            0, 0, 1);    // Translate



    // Calculate final transformation matrix
    Mw2i = Tf * R * M * Sc * T;
//    std::cout <<  "Mw2i = " << std::endl << Mw2i << std::endl;
}
