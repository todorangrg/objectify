#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "utils/base_classes.h"
#include "opencv/cv.h"
#include "visual/plot_data.h"

enum TFmode{
    OLD2NEW,
    NEW2OLD
};

class Segmentation{

public:
    void run(SensorData& sensor, bool advance);
    void plot_data(SensorData& sensor, cv::Scalar color_old, cv::Scalar color_new);

    //Constructors
    Segmentation(RecfgParam &_param, SensorTf& _tf_sns, FrameTf& _tf_frm,PlotData& _plot,KalmanSLDM& _k);

private:
    void sort_seg_init(SegmentDataPtrVectorPtr &segments_init);

    void assign_to_seg_init(const PointDataVectorPtr &_input     , SegmentDataPtrVectorPtr    &segments_init, boost::shared_ptr<std::vector<boost::shared_ptr<ObjectData> > > obj);
    void populate_seg_ext  (const SegmentDataPtrVectorPtr &input , SegmentDataExtPtrVectorPtr &output);
    void calc_tf_vel(SegmentDataPtrVectorPtr &input, SensorData &sensor);
    void calc_tf           (SegmentDataExtPtrVectorPtr &_input, TFmode tf_mode );//OBJECT VEL STUFF
    void calc_occlusion    (SegmentDataExtPtrVectorPtr &_input, int occ_type);//USE ENUM FOR OCC TYPE
    void sample_const_angle(SegmentDataExtPtrVectorPtr &_input);
    void split_segments    (SegmentDataExtPtrVectorPtr &_input);
    void erase_img_outl    (SegmentDataExtPtrVectorPtr &_input);
    void check_neigh_p     (const SegmentDataExtPtrVectorPtr &_input, SegmentDataExtPtrVectorPtr &_temp, std::vector<bool> &temp_valid, IteratorIndexSet iis);
    void calc_com          (SegmentDataPtrVectorPtr &input);

    //Parameters, plot & debug
    double& sensor_range_max;
    double& angle_inc;
    double& angle_max;
    double& angle_min;
    double& segm_discont_dist;

    double& outl_circle_rad;
    double& outl_sigma;
    double& outl_prob_thres;
    SensorTf& tf_sns;
    FrameTf&  tf_frm;
    PlotData& plot;
    bool& plot_data_segm;
    KalmanSLDM    &k;
};

#endif // SEGMENTATION_H


