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


#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include"utils/base_classes.h"

class ConvolInfo{
public:

    int    shift_spl;
    int    it_min_ref;
    int    it_max_ref;

    xy     com_ref;
    xy     com_spl;

    cv::Matx33d T;
    Gauss  ang_distr;
    double sqr_err;
    double pair_no;
    double com_dr;
    double score;

    SegmentDataExtPtr seg_ref;
    SegmentDataExtPtr seg_spl;

    //Constructors & Destructors
    ConvolInfo(int _shift_spl,xy _com_ref, xy _com_spl,Gauss _ang_distr):shift_spl(_shift_spl),ang_distr(_ang_distr),com_ref(_com_ref),com_spl(_com_spl),com_dr(diff(com_ref,com_spl)),sqr_err(0),score(0){}
    ConvolInfo(int _shift_spl,xy _com_ref, xy _com_spl)                 :shift_spl(_shift_spl),                      com_ref(_com_ref),com_spl(_com_spl),com_dr(diff(com_ref,com_spl)),sqr_err(0),score(0){}
    ConvolInfo(int _shift_spl,SegmentDataExtPtr _seg_ref, SegmentDataExtPtr _seg_spl):shift_spl(_shift_spl),seg_ref(_seg_ref),seg_spl(_seg_spl){}
    ~ConvolInfo(){}
};

class AngKeyDataBase{
public:

    int no_keys;
    void init();
    int* add   (double angle);
    int* remove(double angle);

    AngKeyDataBase(double& _d_angl):d_angl(_d_angl){}
private:

    std::vector<int> keys;
    double&          d_angl;
    int normalize_key(int ang_key);
};



class Convolution{
public:

    void runConvolution();

    //Constructors & Destructors
    Convolution(RecfgParam &_param);
    ~Convolution(){}
protected:

    boost::shared_ptr<ConvData>                     conv_data[2];   //ref and spl const_dist sampled point info
    std::vector<boost::shared_ptr<ConvolInfo> >     conv_distr;     //computed convolution positions with info
    std::vector<boost::shared_ptr<ConvolInfo> >     conv_accepted;  //accepted convolution positions with info

    bool create_normal_database(CorrInput& pair);
    bool convolute();
    void fade_out_snapped_p();
private:

    void sample_const_dist(SegmentDataExtPtr& input,ConvolStatus conv_stat);
    void smooth_normals(ConvolStatus conv_stat);

    void set_convol_it_bounds(boost::shared_ptr<ConvolInfo> input);
    void init_link_ang_key_db(boost::shared_ptr<ConvolInfo> input);
    void create_convol_com   (int init_shift,int final_shift,int step);
    void normal_snapp        (boost::shared_ptr<ConvolInfo> input);
    void compute_tf_SVD      (boost::shared_ptr<ConvolInfo> input);
    void compute_tf_ANG_VAR  (boost::shared_ptr<ConvolInfo> input);
    bool find_accepted_tf_zones();
    void add_accepted_tf       (int c_acc_it_min, int c_acc_it_max);

    double snap_score   (boost::shared_ptr<ConvolInfo> input, bool SVD);
    double fade_out_func(const PointDataSample& p_ref, const PointDataSample& p_spl);
    double weight_func  (const PointDataSample& p_ref, const PointDataSample& p_spl);

    AngKeyDataBase ang_key_db;
    double         com_dr_max;

    double&            sample_dist;
    double&            min_len_perc;
    double&            marg_extr_excl;
    std::list<double>& smooth_mask;

    double& com_dr_thres;
    double& ang_mean_thres;
    double& ang_var_thres;
    double& sqr_err_thres;
    double& p_no_perc_thres;
    double& score_thres;
    bool&   full_search;
    bool&   SVD;
    double& noise_ang_base;
};

#endif //CONVOLUTION_H
