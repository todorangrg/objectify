#ifndef CORRELATION_H
#define CORRELATION_H

#include "utils/base_classes.h"
#include "visual/plot_data.h"
#include "visual/plot_convolution.h"
#include "utils/convolution.h"
#include "utils/kalman.h"

#define stringify( name ) # name

enum CorrPairFlag{
    CORR_NO_CORR_NEW,
    CORR_NO_CORR_OLD,
    CORR_SINGLE2,
    CORR_SINGLE1MULTI1,
    CORR_MULTI2
};

class NeighData{
public:
    SegmentDataExtPtr neigh;
    double         prob_fwd;
    double         prob_rev;
    NeighData(SegmentDataExtPtr _neigh, double _prob_fwd, double _prob_rev): neigh(_neigh), prob_fwd(_prob_fwd), prob_rev(_prob_rev){}
};


class Correlation : public Convolution{

public:
    void run(SensorData& sensor);
    void plot_all_data(SensorData& sensor, cv::Scalar color_old, cv::Scalar color_new, KalmanSLDM k);


    //Constructors
    Correlation(RecfgParam &_param,PlotData& _plot,PlotConv& _plot_conv);

    std::map <SegmentDataExtPtr, std::vector<NeighData> > neigh_data[2];

private:
    void update_neigh_list();





    std::vector<CorrInput> corr_list;

    void run_conv(SensorData& sensor);
    void calc_stitch_perc    (const SegmentDataExtPtrVectorPtr &input_ref, const SegmentDataExtPtrVectorPtr &input_spl, FrameStatus fr_status);
    void insert_in_corr_queue(const SegmentDataExtPtr          seg_p_old , const NeighData&         seg_p_new  );
    void insert_in_corr_queue(const SegmentDataExtPtr          seg_p_old);
    void set_flags           (      SegmentDataExtPtrVectorPtr &input_old,       SegmentDataExtPtrVectorPtr &input_new );
    void merge_neigh_lists   (FrameStatus fr_status);
    void resolve_weak_links  (FrameStatus fr_status);
    void create_corr_queue   ();
    CorrPairFlag corr_pair_flag( CorrFlag flag1, CorrFlag flag2 );

    //Parameters, plot & debug
    double& queue_d_thres;
    double& neigh_circle_rad;
    bool&   viz_convol_all;
    bool&   viz_corr_links;
    bool&   viz_data_tf;
    int&    viz_convol_step_no;
    PlotData& plot_data;
    PlotConv& plot_conv;

    void debug_cout_segment   (const SegmentDataExtPtr it_seg,bool old);
    void debug_cout_neigh_list(FrameStatus fr_status);
    void debug_cout_corr_queue(std::vector<CorrInput>& corr_queue);
};

#endif // CORRELATION_H


