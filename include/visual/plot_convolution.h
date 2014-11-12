#ifndef PLOT_CONVOLUTION_H
#define PLOT_CONVOLUTION_H

#include "utils/base_classes.h"
#include "visual/plot.h"
#include "utils/convolution.h"
#include "opencv/cv.h"


//ADD SEGMENT/OBJECT INFO FOR THE REF/SPL FRAME
//ADD CONVOL INFO FOR THE LOOKED-AT POSITION + COVARIANCE (xy:x,y<-EIGEN,teta) OF THE ENTIRE SOLUTION SPACE


class ConvolInfo;

enum PlotConvGraph{
    DR_MEAN,
    P_PAIR,
    SQR_ERR,
    SCORE,
    ANG_VAR
};
struct GraphData{
    std::stringstream name;
    double max;
    double min;
    double* max_thrs;
    double* min_thrs;
    double scale;
    cv::Scalar color;
};

class PlotConv : public Plot{
public:
    void init_w2i(boost::shared_ptr<ConvData>* conv_data);
    void update();

    void plot_conv_info  (std::vector<boost::shared_ptr<ConvolInfo> >& convol_distr,cv::Scalar color_mean,cv::Scalar color_var,cv::Scalar color_dist);
    void plot_conv_points(std::vector<boost::shared_ptr<ConvolInfo> >& convol_distr, boost::shared_ptr<ConvData>* conv_data);
    void plot_grids();
    void load_graph_param(std::vector<boost::shared_ptr<ConvolInfo> >& convol_distr);

    PlotConv(std::string wndView,RecfgParam &_param);
private:



    static const int no_graph_param = 5;

    bool&       plot_conv;
    bool&       plot_normals;
    bool&       plot_tf;
    bool&       tf_ref2spl;

    int&   convol_no;

    double& com_dr_thres;
    double& ang_var_thres;
    double& sqr_err_thres;
    double& score_thres;
    int     image_size;



    double scale_x;

    int    x_border[2];
    int    y_border[2];
    double font_size;

    GraphData gdata[no_graph_param];

    double Mw2i_scale;
};

#endif // PLOT_CONVOLUTION_H
