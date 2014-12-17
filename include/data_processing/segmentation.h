#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "utils/base_classes.h"
#include "opencv/cv.h"
#include "visual/plot_data.h"

enum TFmode{
    OLD2NEW,
    NEW2OLD
};

enum OcclType{
    IN_SEG,
    IN_ALL
};

class Segmentation{
public:

    void run(InputData& input, KalmanSLDM &k, bool advance);
    void plot_data(InputData& input, KalmanSLDM k, cv::Scalar col_seg_oi, cv::Scalar col_seg_oe, cv::Scalar col_seg_ni, cv::Scalar col_seg_ne);

    //Constructors & Destructors
    Segmentation(RecfgParam &_param, SensorTf& _tf_sns, PlotData& _plot,KalmanSLDM& _k);
    ~Segmentation(){}
private:

    bool in_range(polar p);

    void sort_seg_init(SegmentDataPtrVectorPtr &segments_init);

    void assign_seg_init   (const PointDataVectorPtr         &input , SegmentDataPtrVectorPtr    &segments_init);
    void assign_seg_ext    (const SegmentDataPtrVectorPtr    &input , SegmentDataExtPtrVectorPtr &output);
    void link_init_ext     (      SegmentDataExtPtrVectorPtr &ext);

    void split_for_occl    (      SegmentDataExtPtrVectorPtr &input);
    void calc_occlusion    (      SegmentDataExtPtrVectorPtr &_input, OcclType occ_type);
    void sample_const_angle(      SegmentDataExtPtrVectorPtr &_input);
    void erase_img_outl    (      SegmentDataExtPtrVectorPtr &_input);
    void check_neigh_p     (const SegmentDataExtPtrVectorPtr &_input, SegmentDataExtPtrVectorPtr &_temp,
                                       std::vector<bool> &temp_valid, IteratorIndexSet<SegmentDataExt> iis);

    template <class SegData> void calc_tf      (boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, TFmode tf_mode );
    template <class SegData> void split_com_len(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, bool old_init);

    //Parameters, plot & debug
    double& sensor_range_max;
    double& angle_inc;
    double& angle_max;
    double& angle_min;
    double& segm_discont_dist;

    static const int min_seg_dist = 0.2;

    double& outl_circle_rad;
    double& outl_sigma;
    double& outl_prob_thres;
    SensorTf& tf_sns;
    PlotData& plot;
    bool& plot_data_segm_init;
    bool& plot_data_segm_ext;
    KalmanSLDM    &k;

    FrameTf tf_frm;

    FrameTf& get_tf(){ return tf_frm; }
};

#endif // SEGMENTATION_H


