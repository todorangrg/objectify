#ifndef CORRELATION_H
#define CORRELATION_H

#include "utils/base_classes.h"
#include "visual/plot_data.h"
#include "visual/plot_convolution.h"
#include "utils/convolution.h"
#include "utils/kalman.h"

#define stringify( name ) # name





class Correlation : public Convolution{
public:

    std::map <SegmentDataPtr   , std::vector<NeighDataInit> > neigh_data_init[2];
    std::map <SegmentDataExtPtr, std::vector<NeighDataExt > > neigh_data_ext [2];

    void run(InputData& input, KalmanSLDM k, bool new_frame);
    void plot_all_data(InputData& input, KalmanSLDM k, cv::Scalar color_old, cv::Scalar color_new);

    //Constructors & Destructors
    Correlation(RecfgParam &_param,PlotData& _plot,PlotConv& _plot_conv);
    ~Correlation(){}
private:

    void update_neigh_list();

    std::vector<CorrInput> corr_list;

    void run_conv(InputData& input, KalmanSLDM k);
    void calc_stitch_perc    (const SegmentDataPtrVectorPtr &input_ref, const SegmentDataPtrVectorPtr &input_spl, FrameStatus fr_status);
    void insert_in_corr_queue(const SegmentDataExtPtr          seg_p_old , const NeighDataExt&         seg_p_new  );
    void insert_in_corr_queue(const SegmentDataExtPtr          seg_p_old);
    void set_flags           (SegmentDataExtPtrVectorPtr &input_old, FrameStatus fr_stat);
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


    std::string print_segment(SegmentDataBase& it_seg);
    void        print_neigh_list(FrameStatus fr_status);
};

#endif // CORRELATION_H


