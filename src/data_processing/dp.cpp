#include "data_processing/dp.h"
#include "utils/iterators.h"


using namespace cv;

extern const char* CorrFlagNames[];

DataProcessing::DataProcessing(RecfgParam& param,SensorTf& _tf_sns,PlotData& plot,PlotConv& plot_conv) :
    param(param),
    sensor(param),
    preprocessing(param,plot),
    segmentation(param,_tf_sns,sensor.get_tf(),plot, k),
    correlation(param,plot,plot_conv),
    tf_sns(_tf_sns),
    plot(plot){


    State dummy_rob;
    dummy_rob.set(0,0,0);
    k.init(dummy_rob);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void DataProcessing::run(bool new_frame){
    ros::Time t0;

    //advance kalman

    t0 = ros::Time::now();
    preprocessing.run(sensor);
    plot.putInfoText("[ms]Preproc run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,4,plot.black);//does not freeze value in step sim mode
    t0 = ros::Time::now();
    segmentation.run(sensor,new_frame);
    plot.putInfoText("[ms]Segment run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,6,plot.black);//does not freeze value in step sim mode
    t0 = ros::Time::now();
    correlation.run(sensor);
    plot.putInfoText("[ms]Innov run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,8,plot.black);//does not freeze value in step sim mode

    t0 = ros::Time::now();
    correlation   .plot_all_data(sensor,plot.blue_bright,plot.orange, k);
    plot.putInfoText("[ms]Innov plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,9,plot.black);//does not freeze value in step sim mode

//    if(new_frame){
        t0 = ros::Time::now();
        k.run(sensor, correlation.neigh_data[FRAME_OLD], correlation.neigh_data[FRAME_NEW]);
        plot.putInfoText("[ms]Kalman run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,10,plot.black);//does not freeze value in step sim mode
//    }

    t0 = ros::Time::now();
    preprocessing.plot_data(sensor,plot.gray,plot.red,plot.red,plot.green);
    plot.putInfoText("[ms]Preproc plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,5,plot.black);//does not freeze value in step sim mode
    t0 = ros::Time::now();
    segmentation .plot_data(sensor, plot.blue, plot.red);
    plot.putInfoText("[ms]Segment plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,7,plot.black);//does not freeze value in step sim mod

    if(sensor.status == OLD_FRAME){
        t0 = ros::Time::now();
        plot.plot_kalman(sensor.frame_new->seg_ext,k);
        plot.putInfoText("[ms]Kalman plot run time = ",(ros::Time::now() - t0).toNSec()* 1e-6,11,plot.black);//does not freeze value in step sim mode
    }




}

/////------------------------------------------------------------------------------------------------------------------------------------------------///
