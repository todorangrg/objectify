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

    void run(InputData& input, KalmanSLDM k);
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


