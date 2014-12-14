#include "data_processing/dp.h"
#include "utils/iterators.h"
#include "fstream"


using namespace cv;

extern const char* CorrFlagNames[];

DataProcessing::DataProcessing(RecfgParam& param,SensorTf& _tf_sns,PlotData& plot,PlotConv& plot_conv) :
    param(param),
    preprocessing(param,plot),
    k(param, _tf_sns),
    segmentation(param,_tf_sns, plot, k),
    correlation(param,plot,plot_conv),
    tf_sns(_tf_sns),
    plot(plot){


    RState dummy_rob(0,0,0);
    k.init(dummy_rob);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void DataProcessing::run(bool new_frame){
    ros::Time t0;
    std::stringstream info;

    int p_size = 0;
    if(input.sensor_raw){ p_size += input.sensor_raw->size(); }
    info.str(""); info<<"[ n]raw_frames no_p = "<< p_size;    plot.putInfoText(info,0,plot.black);
    info.str(""); info<<"[ms]delta t_frame   = "<<input.u.dt; plot.putInfoText(info,0,plot.black);
    static int i=0;
    if(i < 10){
        i++;
        return;
    }

    k.prediction(k.seg_init, input.u);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t0 = ros::Time::now();
    preprocessing.run(input);
    info.str(""); info<<"[ms]Preproc run time      = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    preprocessing.plot_data(input,plot.gray,plot.red,plot.red,plot.green);
    info.str(""); info<<"[ms]Preproc plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t0 = ros::Time::now();
    segmentation.run(input, k, new_frame);
    info.str(""); info<<"[ms]Segment run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    segmentation .plot_data(input, k);
    info.str(""); info<<"[ms]Segment plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mod

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t0 = ros::Time::now();
    correlation.run(input, k, new_frame);
    info.str(""); info<<"[ms]Innov run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    correlation   .plot_all_data(input, k ,plot.blue_bright,plot.orange);
    info.str(""); info<<"[ms]Innov plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t0 = ros::Time::now();
    k.run(input, correlation.neigh_data_ext[FRAME_OLD], correlation.neigh_data_ext[FRAME_NEW], correlation.neigh_data_init[FRAME_OLD], correlation.neigh_data_init[FRAME_NEW]);
    info.str(""); info<<"[ms]Kalman run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    plot.plot_kalman(k.seg_init,k);
    info.str(""); info<<"[ms]Kalman plot run time = "<<(ros::Time::now() - t0).toNSec()* 1e-6; plot.putInfoText(info,0,plot.black);//does not freeze value in step sim mod

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

/////------------------------------------------------------------------------------------------------------------------------------------------------///
