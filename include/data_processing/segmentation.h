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
    void run(InputData& input, KalmanSLDM &k, bool advance);
    void plot_data(InputData& input,KalmanSLDM k, cv::Scalar color_old, cv::Scalar color_new);

    //Constructors
    Segmentation(RecfgParam &_param, SensorTf& _tf_sns, PlotData& _plot,KalmanSLDM& _k);

private:

    bool in_range(polar p);

    void inform_init_to_ext(SegmentDataPtrVectorPtr &init, SegmentDataExtPtrVectorPtr &ext, bool old_frame);



    void sort_seg_init(SegmentDataPtrVectorPtr &segments_init);

    void assign_seg_init   (const PointDataVectorPtr &_input     , SegmentDataPtrVectorPtr    &segments_init);
    void assign_seg_ext    (const SegmentDataPtrVectorPtr &input , SegmentDataExtPtrVectorPtr &output);
    void assign_seg_init_tf(SegmentDataPtrVectorPtr &input);
    void calc_tf           (SegmentDataExtPtrVectorPtr &_input, TFmode tf_mode );//OBJECT VEL STUFF

    void calc_tf_init(SegmentDataPtrVectorPtr &input, TFmode tf_mode );

    void split_segments_for_occl(SegmentDataExtPtrVectorPtr &input);

    void calc_occlusion    (SegmentDataExtPtrVectorPtr &_input, int occ_type);//USE ENUM FOR OCC TYPE
    void sample_const_angle(SegmentDataExtPtrVectorPtr &_input);
    void split_segments    (SegmentDataExtPtrVectorPtr &_input);
    void erase_img_outl    (SegmentDataExtPtrVectorPtr &_input);
    void check_neigh_p     (const SegmentDataExtPtrVectorPtr &_input, SegmentDataExtPtrVectorPtr &_temp, std::vector<bool> &temp_valid, IteratorIndexSet iis);
    void calc_com          (SegmentDataPtrVectorPtr &input);
    void calc_com_ext(SegmentDataExtPtrVectorPtr &input);

    void compute_len_init(SegmentDataPtrVectorPtr &input);

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
    PlotData& plot;
    bool& plot_data_segm;
    KalmanSLDM    &k;

    FrameTf tf_frm;

    FrameTf& get_tf(){ return tf_frm; }
};

#endif // SEGMENTATION_H


