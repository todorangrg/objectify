#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "utils/base_classes.h"

class PlotData;

class Preprocessing{

public:
    void run(InputData& input);
    void plot_data(InputData& input, cv::Scalar color_raw, cv::Scalar color_preproc, cv::Scalar color_outl_acc,cv::Scalar color_outl_rej);

    //Constructors
    Preprocessing(RecfgParam &_param, PlotData& _plot);

private:
    void filter       (const PointDataVectorPtr &_input, PointDataVectorPtr &_output);
    void erase_outl   (const PointDataVectorPtr &_input, PointDataVectorPtr &_output, bool debug_print);
    void check_neigh_p(const PointDataVectorPtr &_input, PointDataVectorPtr   &_temp, std::vector<bool> &temp_valid, int it, bool debug_print);

    //Parameters, plot & debug
    double& angle_inc;

    bool&   filter_input;
    double& filter_circle_rad;
    bool&   filter_circle_rad_scale;
    double& filter_sigma;

    double& outl_circle_rad;
    double& outl_sigma;
    double& outl_prob_thres;

    PlotData&   plot;
    bool& plot_data_raw;
    bool& plot_data_preproc;
    bool& plot_oult_preproc;
    std::vector<cv::RotatedRect> plot_p_ell;
    std::vector<int>             plot_p_ell_erased;
};

#endif // PREPROCESSING_H
