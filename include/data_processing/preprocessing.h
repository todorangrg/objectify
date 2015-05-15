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


#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "utils/base_classes.h"

class PlotData;

class Preprocessing{
public:

    void run(InputData& input);
    void plot_data(InputData& input, cv::Scalar color_raw, cv::Scalar color_preproc, cv::Scalar color_outl_acc,cv::Scalar color_outl_rej);

    //Constructors & Destructors
    Preprocessing(RecfgParam &_param, PlotData& _plot);
    ~Preprocessing(){}
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
