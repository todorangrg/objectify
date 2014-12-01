#include "data_processing/dp.h"
#include "utils/iterators.h"


using namespace cv;

extern const char* CorrFlagNames[];

DataProcessing::DataProcessing(RecfgParam& param,SensorTf& _tf_sns,PlotData& plot,PlotConv& plot_conv) :
    param(param),
    preprocessing(param,plot),
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

    //advance kalman in main
    k.prediction(k.seg_init, input.u);
    //k.predict(v robot, dt)   make it bool, when not possible loop without doing anything

    t0 = ros::Time::now();
    preprocessing.run(input);
    plot.putInfoText("[ms]Preproc run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,4,plot.black);//does not freeze value in step sim mode
    t0 = ros::Time::now();
    segmentation.run(input, k, new_frame);
    plot.putInfoText("[ms]Segment run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,6,plot.black);//does not freeze value in step sim mode


    t0 = ros::Time::now();
    correlation.run(input, k, new_frame);
    plot.putInfoText("[ms]Innov run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,8,plot.black);//does not freeze value in step sim mode


    t0 = ros::Time::now();
    segmentation .plot_data(input, k, plot.blue, plot.red);
    plot.putInfoText("[ms]Segment plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,7,plot.black);//does not freeze value in step sim mod


    t0 = ros::Time::now();
    correlation   .plot_all_data(input, k ,plot.blue_bright,plot.orange);
    plot.putInfoText("[ms]Innov plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,9,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    //k.run(sensor, correlation.neigh_data[FRAME_OLD], correlation.neigh_data[FRAME_NEW]);
    k.run(input, correlation.neigh_data_ext[FRAME_OLD], correlation.neigh_data_ext[FRAME_NEW], correlation.neigh_data_init[FRAME_OLD]);
    //k.update......
    plot.putInfoText("[ms]Kalman run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,10,plot.black);//does not freeze value in step sim mode


    t0 = ros::Time::now();
    preprocessing.plot_data(input,plot.gray,plot.red,plot.red,plot.green);
    plot.putInfoText("[ms]Preproc plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,5,plot.black);//does not freeze value in step sim mode



//    if(k.status == OLD_FRAME){
        t0 = ros::Time::now();
        plot.plot_kalman(input.seg_ext,k);
        plot.putInfoText("[ms]Kalman plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,11,plot.black);//does not freeze value in step sim mode
//    }




}

/////------------------------------------------------------------------------------------------------------------------------------------------------///
