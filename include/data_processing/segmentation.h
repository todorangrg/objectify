/***************************************************************************
 *   Software License Agreement (BSD License)                              *
 *   Copyright (C) 2015 by Horatiu George Todoran <todorangrg@gmail.com>   *
 *                                                                         *
 *   Redistribution and use in source and binary forms, with or without    *
 *   modification, are permitted provided that the following conditions    *
 *   are met:                                                              *
 *                                                                         *
 *   1. Redistributions of source code must retain the above copyright     *
 *      notice, this list of conditions and the following disclaimer.      *
 *   2. Redistributions in binary form must reproduce the above copyright  *
 *      notice, this list of conditions and the following disclaimer in    *
 *      the documentation and/or other materials provided with the         *
 *      distribution.                                                      *
 *   3. Neither the name of the copyright holder nor the names of its      *
 *      contributors may be used to endorse or promote products derived    *
 *      from this software without specific prior written permission.      *
 *                                                                         *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   *
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     *
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS     *
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE        *
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,  *
 *   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
 *   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      *
 *   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      *
 *   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT    *
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY *
 *   WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           *
 *   POSSIBILITY OF SUCH DAMAGE.                                           *
 ***************************************************************************/


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
    void run_future(SegmentDataPtrVectorPtr & seg_init, SegmentDataExtPtrVectorPtr & seg_ext_now, SegmentDataExtPtrVectorPtr & seg_ext_ftr, FrameTf & tf, RState rob_f0, KInp u, KalmanSLDM &k);


    void plot_data(InputData& input, KalmanSLDM k, cv::Scalar col_seg_oi, cv::Scalar col_seg_oe, cv::Scalar col_seg_ni, cv::Scalar col_seg_ne);

    //Constructors & Destructors
    Segmentation(RecfgParam &_param, SensorTf& _tf_sns, PlotData& _plot,KalmanSLDM& _k);
    ~Segmentation(){}


    template <class SegData> void calc_tf      (boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, TFmode tf_mode, FrameTf _tf_frm);
    void calc_seg_tf(PointDataVector & _p, TFmode tf_mode, FrameTf _tf_frm);

    double& sensor_range_max;
    double& angle_max;
    double& angle_min;
private:

    bool in_range(polar p);

    void sort_seg_init(SegmentDataPtrVectorPtr &segments_init);

    void assign_seg_init   (const PointDataVectorPtr         &input , SegmentDataPtrVectorPtr    &segments_init);
    void assign_seg_ext    (const SegmentDataPtrVectorPtr    &input , SegmentDataExtPtrVectorPtr &output, bool in_rangee);
    void link_init_ext     (      SegmentDataExtPtrVectorPtr &ext);

    void split_for_occl    (      SegmentDataExtPtrVectorPtr &input);
    void calc_occlusion    (      SegmentDataExtPtrVectorPtr &_input, OcclType occ_type);
    void sample_const_angle(      SegmentDataExtPtrVectorPtr &_input);
    void erase_img_outl    (      SegmentDataExtPtrVectorPtr &_input);
    void check_neigh_p     (const SegmentDataExtPtrVectorPtr &_input, SegmentDataExtPtrVectorPtr &_temp,
                                       std::vector<bool> &temp_valid, IteratorIndexSet<SegmentDataExt> iis);


    template <class SegData> void split_com_len(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, bool old_init);


    void bloat_image(SegmentDataExtPtrVectorPtr &_input, double radius);


    //Parameters, plot & debug

    double& angle_inc;

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


