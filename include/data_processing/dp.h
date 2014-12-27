#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <ros/ros.h>
#include "utils/base_classes.h"
#include "utils/math.h"
#include "data_processing/preprocessing.h"
#include "data_processing/segmentation.h"
#include "data_processing/correlation.h"
#include "visual/plot_data.h"
#include "utils/kalman.h"
#include "planner/planner.h"

class DataProcessing{
public:

    InputData     input;
    Preprocessing preprocessing;
    Segmentation  segmentation;
    Correlation   correlation;
    KalmanSLDM    k;
    TangentBug    planner;

    void run(bool new_frame);

    //Constructors & Destructors
    DataProcessing(RecfgParam& param, SensorTf& _tf_sns, PlotData& plot, PlotConv& plot_conv , rosbag::Bag &bag);
    ~DataProcessing(){}
protected:

    RecfgParam &param;
    Distributions noise;
    SensorTf& tf_sns;
    PlotData& plot;
private:

};

#endif // DATA_PROCESSING_H
