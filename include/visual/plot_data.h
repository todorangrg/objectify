#ifndef PLOT_DATA_H
#define PLOT_DATA_H

#include "utils/base_classes.h"
#include "visual/plot.h"


class ConvolInfo;
class KalmanSLDM;

class PlotData : public Plot{
public:
    void init_w2i();
    void update();

    void putInfoText      (const cv::string &text, double var, int row, cv::Scalar color);
    void plot_points      (const PointDataVectorPtr &data     , cv::Scalar color);
    void plot_segm        (SegmentDataExtPtrVectorPtr &data, cv::Scalar color);
    void plot_segm_init(SegmentDataPtrVectorPtr &data, cv::Scalar color);
    void plot_corr_links  (const std::vector<CorrInput>& list , cv::Scalar color_o2n,cv::Scalar color_n2o);
    void plot_oult_circles(const std::vector<cv::RotatedRect>& plot_p_ell, std::vector<int>& plot_p_ell_erased, cv::Scalar color_acc,cv::Scalar color_rej);

    void plot_segm_tf(const SegmentDataExtPtrVectorPtr &data, int frame, cv::Scalar color);

    void plot_kalman(const SegmentDataExtPtrVectorPtr &data, KalmanSLDM& k);




    PlotData(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns);
    SensorTf&   tf_sns;

    int&        image_size;
private:
    bool&       plot_data;

    bool&       plot_grid;
    double&     sensor_range_max;
    double&     angle_max;
    double&     angle_min;


};

#endif // PLOT_DATA_H
