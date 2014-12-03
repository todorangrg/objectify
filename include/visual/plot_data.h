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

    void putInfoText      (std::stringstream &text_data, int row, cv::Scalar color);
    void plot_points      (PointDataVector &data     , cv::Scalar color);

    template <class SegData>
    void plot_segm        (boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, cv::Scalar color);


    void plot_segm_init   (SegmentDataPtrVectorPtr &data, cv::Scalar color);
    void plot_corr_links  (const std::vector<CorrInput>& list , cv::Scalar color_o2n,cv::Scalar color_n2o);
    void plot_oult_circles(const std::vector<cv::RotatedRect>& plot_p_ell, std::vector<int>& plot_p_ell_erased, cv::Scalar color_acc,cv::Scalar color_rej);

    void plot_segm_tf(const SegmentDataExtPtrVectorPtr &data, int frame, cv::Scalar color);

    void plot_kalman(const SegmentDataExtPtrVectorPtr &data, KalmanSLDM& k);

    //Constructors & Destructors
    PlotData(std::string wndView,RecfgParam &_param, SensorTf& _tf_sns);
    ~PlotData(){}

    SensorTf&   tf_sns;
    int&        image_size;
private:
    bool        text[50];

    bool&       plot_data;

    bool&       plot_grid;
    double&     sensor_range_max;
    double&     angle_max;
    double&     angle_min;
};

extern template void PlotData::plot_segm<SegmentData   >(SegmentDataPtrVectorPtr    &data, cv::Scalar color);
extern template void PlotData::plot_segm<SegmentDataExt>(SegmentDataExtPtrVectorPtr &data, cv::Scalar color);


#endif // PLOT_DATA_H
