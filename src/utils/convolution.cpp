#include "utils/base_classes.h"
#include "utils/convolution.h"
#include "utils/math.h"
#include <algorithm>

Convolution::Convolution(RecfgParam &_param):
    full_search(_param.convol_full_search),

    SVD(_param.convol_SVD),

    sample_dist(_param.convol_sample_dist),
    marg_extr_excl(_param.convol_marg_extr_excl),
    smooth_mask(_param.smooth_mask),
    ang_key_db(_param.convol_key_d_angle),

    com_dr_thres(_param.convol_com_dr_thres),
    ang_mean_thres(_param.convol_ang_mean_thres),
    ang_var_thres(_param.convol_ang_var_thres),
    sqr_err_thres(_param.convol_sqr_err_thres),
    p_no_perc_thres(_param.convol_p_no_perc_thres),
    score_thres(_param.convol_score_thres){}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::runConvolution(){

}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool Convolution::create_normal_database(CorrInput& pair){
    if(pair.reverse){
        sample_const_dist(pair.frame_new,CONV_REF);
        sample_const_dist(pair.frame_old,CONV_SPL);
    }
    else{
        sample_const_dist(pair.frame_new,CONV_SPL);
        sample_const_dist(pair.frame_old,CONV_REF);
    }
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::fade_out_snapped_p(){
    for(int i=0; i < conv_accepted.size(); i++){
        for(int j = conv_accepted[i]->it_min_ref; j < conv_accepted[i]->it_max_ref; j++){
            conv_data[CONV_REF]->p_cd->at(j                            ).fade_out+= 1.0 / (double)conv_accepted.size();
            conv_data[CONV_SPL]->p_cd->at(j+conv_accepted[i]->shift_spl).fade_out+= 1.0 / (double)conv_accepted.size();
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool d_sort_func    (boost::shared_ptr<ConvolInfo> i, boost::shared_ptr<ConvolInfo> j) { return (i->com_dr    < j->com_dr ); }
bool shift_sort_func(boost::shared_ptr<ConvolInfo> i, boost::shared_ptr<ConvolInfo> j) { return (i->shift_spl < j->shift_spl); }

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::convolute(){
    if(full_search){ com_dr_max = 0.0; }
    else           { com_dr_max = com_dr_thres; }
    int init_shift  =  - p_no_perc_thres * conv_data[CONV_REF]->p_cd->size();
    int final_shift =  - init_shift + conv_data[CONV_SPL]->p_cd->size() - conv_data[CONV_REF]->p_cd->size();
    conv_distr   .clear();
    conv_accepted.clear();
    if(final_shift - init_shift < 1 ){
        return;
    }
    conv_distr   .reserve(final_shift - init_shift);
    conv_accepted.reserve(final_shift - init_shift);

    create_convol_com(init_shift, final_shift, 1);//sort convol pos based on calc com
    std::sort(conv_distr.begin(), conv_distr.end(), d_sort_func);

    for(std::vector<boost::shared_ptr<ConvolInfo> >::iterator it = conv_distr.begin(); it!= conv_distr.end(); it++  ){
        normal_snapp(*it);
        //stops search after max dist was reached.....
        if((com_dr_max < (*it)->com_dr)&&(!full_search)){
            conv_distr.erase(++it,conv_distr.end());
            break;
        }
        //--stops search after max dist was reached.....
    }
    std::sort(conv_distr.begin(), conv_distr.end(), shift_sort_func);
    std::sort(conv_accepted.begin(), conv_accepted.end(), shift_sort_func);

    find_accepted_tf_zones();
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::create_convol_com(int init_shift,int final_shift,int step){///TODO: make it less messy
    xy com_ref(0,0);
    xy com_spl(0,0);
    conv_distr.push_back(boost::shared_ptr<ConvolInfo>(new ConvolInfo(init_shift,conv_data[CONV_REF]->seg,conv_data[CONV_SPL]->seg)));
    set_convol_it_bounds(conv_distr.back());
    for(int i = conv_distr.back()->it_min_ref; i < conv_distr.back()->it_max_ref; i += step){
        com_ref += to_xy(conv_data[CONV_REF]->p_cd->at(i));
        com_spl += to_xy(conv_data[CONV_SPL]->p_cd->at(i + conv_distr.back()->shift_spl));
    }
    double it_size = conv_distr.back()->it_max_ref - conv_distr.back()->it_min_ref;
    conv_distr.back()->com_ref = xy(com_ref.x/it_size/(double)step,com_ref.y/it_size/(double)step);
    conv_distr.back()->com_spl = xy(com_spl.x/it_size/(double)step,com_spl.y/it_size/(double)step);
    conv_distr.back()->com_dr  = diff(conv_distr.back()->com_ref, conv_distr.back()->com_spl);

    if(( conv_distr.back()->com_dr > com_dr_max )&&(full_search)){
        com_dr_max = conv_distr.back()->com_dr;
    }
    for(int shift = init_shift + step; shift < final_shift; shift += step){
        conv_distr.push_back(boost::shared_ptr<ConvolInfo>(new ConvolInfo(shift,conv_data[CONV_REF]->seg,conv_data[CONV_SPL]->seg)));
        set_convol_it_bounds(conv_distr.back());
        for(int i = conv_distr.back()->it_max_ref; i < conv_distr[conv_distr.size() - 2]->it_max_ref; i += step){
            com_ref -= to_xy(conv_data[CONV_REF]->p_cd->at(i));
        }
        for(int i = conv_distr.back()->it_min_ref; i < conv_distr[conv_distr.size() - 2]->it_min_ref; i += step){
            com_ref += to_xy(conv_data[CONV_REF]->p_cd->at(i));
        }
        for(int i = conv_distr[conv_distr.size() - 2]->it_max_ref + conv_distr[conv_distr.size() - 2]->shift_spl; i < conv_distr.back()->it_max_ref + conv_distr.back()->shift_spl; i += step){
            com_spl += to_xy(conv_data[CONV_SPL]->p_cd->at(i));
        }
        for(int i = conv_distr[conv_distr.size() - 2]->it_min_ref + conv_distr[conv_distr.size() - 2]->shift_spl; i < conv_distr.back()->it_min_ref + conv_distr.back()->shift_spl; i += step){
            com_spl -= to_xy(conv_data[CONV_SPL]->p_cd->at(i));
        }
        it_size = conv_distr.back()->it_max_ref - conv_distr.back()->it_min_ref;
        conv_distr.back()->com_ref = xy(com_ref.x/it_size/(double)step,com_ref.y/it_size/(double)step);
        conv_distr.back()->com_spl = xy(com_spl.x/it_size/(double)step,com_spl.y/it_size/(double)step);
        conv_distr.back()->com_dr  = diff(conv_distr.back()->com_ref, conv_distr.back()->com_spl);

        if(( conv_distr.back()->com_dr > com_dr_max )&&(full_search)){
            com_dr_max = conv_distr.back()->com_dr;
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::set_convol_it_bounds(boost::shared_ptr<ConvolInfo> input){
    double it_min_ref = std::max(- input->shift_spl, 0);
    double it_max_ref = std::min(conv_data[CONV_REF]->p_cd->size(), conv_data[CONV_SPL]->p_cd->size() - input->shift_spl);
    input->it_min_ref = it_min_ref + round((it_max_ref - it_min_ref) * marg_extr_excl  /  2.0);
    input->it_max_ref = it_max_ref - round((it_max_ref - it_min_ref) * marg_extr_excl  /  2.0);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::normal_snapp(boost::shared_ptr<ConvolInfo> input){
    set_convol_it_bounds(input);
    init_link_ang_key_db(input);

    input->pair_no = 0;
    if(SVD){
        compute_tf_SVD(input);
    }
    else{
        compute_tf_ANG_VAR(input);
    }
    if( input->score > 0.0 ){
        conv_accepted.push_back(input);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::init_link_ang_key_db(boost::shared_ptr<ConvolInfo> input){
    ang_key_db.init();
    for(int i = input->it_min_ref; i < input->it_max_ref; i++){
        conv_data[CONV_REF]->p_cd->at(i).no = ang_key_db.add(conv_data[CONV_REF]->p_cd->at(i).normal_ang);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::compute_tf_SVD(boost::shared_ptr<ConvolInfo> input){
    Gauss distr;
    input->pair_no = 0;

    cv::Matx<double, 2, 1> m_com_ref(input->com_ref.x, input->com_ref.y);
    cv::Matx<double, 2, 1> m_com_spl(input->com_spl.x, input->com_spl.y);
    cv::Matx<double, 2, 2> m_H;
    m_H = m_H.zeros();

    double weight_sum = 0, weight_sqr_sum = 0;
    for(int i = input->it_min_ref; i < input->it_max_ref; i++){
        xy pi_ref = to_xy(conv_data[CONV_REF]->p_cd->at(i));
        xy pi_spl = to_xy(conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl));
        cv::Matx<double, 2, 1> m_pi_ref(pi_ref.x, pi_ref.y);
        cv::Matx<double, 2, 1> m_pi_spl(pi_spl.x, pi_spl.y);

        m_H = m_H + (m_pi_ref - m_com_ref) * (m_pi_spl - m_com_spl).t() * weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl));
        weight_sum += weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl));
        weight_sqr_sum += sqr(weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl)));
        input->pair_no += /*1.0;*/fade_out_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i+ input->shift_spl));//input pair_no insertion   !!!!!!!!!!!!!!!!!!!!!!!!!!
    }
    m_H(0,0) *=  weight_sum / ( sqr(weight_sum) - weight_sqr_sum ) ;
    m_H(0,1) *=  weight_sum / ( sqr(weight_sum) - weight_sqr_sum ) ;
    m_H(1,0) *=  weight_sum / ( sqr(weight_sum) - weight_sqr_sum ) ;
    m_H(1,1) *=  weight_sum / ( sqr(weight_sum) - weight_sqr_sum ) ;
    cv::SVD svd(m_H, cv::SVD::FULL_UV);
    cv::Mat Rr = svd.vt.t() * svd.u.t();
    if(fabs(cv::determinant(Rr) + 1.0) < 0.0001) {
        svd.vt.row(1) *= -1;
        Rr = svd.vt.t() * svd.u.t();
    }
    cv::Matx<double, 2, 2> R(Rr);
    cv::Matx<double, 2, 1> T = - R * m_com_ref + m_com_spl;
    cv::Matx<double,3,3> R3 (R(0,0), R(0,1),      0, R(1,0), R(1,1),      0, 0, 0, 1);
    cv::Matx<double,3,3> T3 (     1,      0, T(0,0),      0,      1, T(0,1), 0, 0, 1);
    input->T = T3 * R3;                                     ///input tf insertion

    double sqr_err=0;
    for(int i = input->it_min_ref; i < input->it_max_ref; i++){
        sqr_err += sqr( diff( mat_mult(input->T, to_xy(conv_data[CONV_REF]->p_cd->at(i))), to_xy(conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl)))) * weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl));;

        double ang = conv_data[CONV_REF]->p_cd->at(i).normal_ang - conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl).normal_ang;  normalizeAngle(ang);
        distr.add_w_sample(ang, weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl)));
    }
    if(sqr_err != 0.0){
        sqr_err *= weight_sum / ( sqr(weight_sum) - weight_sqr_sum ) ;///=(double)(input->it_max_ref - input->it_min_ref);
    }
    input->sqr_err   = sqr_err;                            ///input sqr_err insertion
    input->ang_distr = distr;                              ///input angular distribution insertion
    input->score     = snap_score(input, /*SVD mode*/true);///input score insertion
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::compute_tf_ANG_VAR(boost::shared_ptr<ConvolInfo> input){
    Gauss distr;
    input->pair_no = 0;

    for(int i = input->it_min_ref; i < input->it_max_ref; i++){
        double ang = conv_data[CONV_REF]->p_cd->at(i).normal_ang - conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl).normal_ang;  normalizeAngle(ang);
        distr.add_w_sample(ang, weight_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl)));
        input->pair_no += /*1.0;*/fade_out_func(conv_data[CONV_REF]->p_cd->at(i), conv_data[CONV_SPL]->p_cd->at(i+input->shift_spl)); //!!!!!!!!!!!!!!!!!!!!!!!!!!
//            if( distr.getVariance() > ang_var_thres + 0.1 ){//ninja break if variance is big => no match
//                break;
//            }
    }
    input->ang_distr = distr;                                   ///input angular distribution insertion
    if( distr.getVariance() < ang_var_thres ){//here conditions for accepting the correlated images in good, covarianceble stuff
        cv::Matx<double, 3, 1> m_com_ref(input->com_ref.x, input->com_ref.y, 1);
        cv::Matx<double, 3, 1> m_com_spl(input->com_spl.x, input->com_spl.y, 1);
        set_tf_mat(input->T, xy(0,0), - distr.getMean());
        cv::Matx<double, 3, 1> T;
        T = - input->T * m_com_ref + m_com_spl;
        set_tf_mat(input->T,xy(T(0,0),T(0,1)),-distr.getMean());///input tf insertion
    }
    double sqr_err=0;//TODO WEIGHT THE SQR ERROR
    for(int i = input->it_min_ref; i < input->it_max_ref; i++){
        sqr_err += sqr( diff( mat_mult(input->T, to_xy(conv_data[CONV_REF]->p_cd->at(i))), to_xy(conv_data[CONV_SPL]->p_cd->at(i + input->shift_spl))));
    }
    if(sqr_err != 0.0){
        sqr_err/=(double)(input->it_max_ref - input->it_min_ref);
    }
    input->sqr_err = sqr_err;                                  ///input sqr_err insertion
    input->score = snap_score(input, /*ANG_VAR mode*/false);   ///input score insertion
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

double Convolution::snap_score(boost::shared_ptr<ConvolInfo> input, bool _SVD){//used by SVD-snapp mode ; 1 = perfect match, 0.0 = no match
    double err_score, occl_score, com_d_score, rot_score;
    if(_SVD){ err_score = (1.0 - input->sqr_err / sqr_err_thres);            }                     //best case = 1.0 , worst case < 0.0
    else    { err_score = (1.0 - input->ang_distr.getVariance() / ang_var_thres);}
    occl_score  = (1.0 - input->pair_no / (double)conv_data[CONV_REF]->p_cd->size()) / p_no_perc_thres; //best case = 0.0 , worst case > 1.0
    com_d_score = input->com_dr / com_dr_max;                                                      //best case = 0.0 , worst case > 1.0
    rot_score   = fabs(input->ang_distr.getMean() / ang_mean_thres);                                  //best case = 0.0 , worst case > 1.0

    if(( err_score < 0.0)||(occl_score > 1.0)||(com_d_score > 1.0)||(rot_score > 1.0)){
        return 0.0;
    }
    //return fmax(0.0, err_score - (1.0 / 3.0) * (occl_score + com_d_score + rot_score));
    return fmax(-1, err_score - /*sqr*/(1.0-err_score)*((1.0 - occl_score) * sqr(1.0 - com_d_score) * sqr(1.0 -  rot_score)));
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

double Convolution::fade_out_func(const PointDataSample& p_ref, const PointDataSample& p_spl){
    double fade_out = fmax(p_ref.fade_out, p_spl.fade_out);
    return fmax(1.0 - fade_out, 0.0);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

double Convolution::weight_func(const PointDataSample& p_ref, const PointDataSample& p_spl){
    return (1.0 / ((double) *p_ref.no * ang_key_db.no_keys) ) * fade_out_func(p_ref, p_spl);// !!!!!!!!!!!!!!!!!!!!!!!!!!
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

bool best_score_first_sort_func(TFdata i, TFdata j) { return (i.score > j.score ); }

void Convolution::find_accepted_tf_zones(){
    bool new_tf_zone = true;
    int i=0, j=0;
    int acc_it_min = 0, acc_it_max = 0;
    while((i < conv_accepted.size()) && (j < conv_distr.size())){
        while(( conv_distr[j] != conv_accepted[i] )&&(j < conv_distr.size())){ j++;new_tf_zone = true; }
        acc_it_min = i;
        while(( conv_distr[j] == conv_accepted[i] )&&(i < conv_accepted.size())){ j++;i++; }
        acc_it_max = i;
        if(new_tf_zone){
            add_accepted_tf(acc_it_min, acc_it_max);
        }
        new_tf_zone = false;
    }
    ////////////////////////////////////////STORES ONLY BEST MATCH
    if(!full_search){
        if(conv_data[CONV_REF]->tf->size() > 1 ){
            std::sort(conv_data[CONV_REF]->tf->begin(), conv_data[CONV_REF]->tf->end(), best_score_first_sort_func);
            conv_data[CONV_REF]->tf->erase(conv_data[CONV_REF]->tf->begin()+1, conv_data[CONV_REF]->tf->end());
        }
        if(conv_data[CONV_SPL]->tf->size() > 1 ){
            std::sort(conv_data[CONV_SPL]->tf->begin(), conv_data[CONV_SPL]->tf->end(), best_score_first_sort_func);
            conv_data[CONV_SPL]->tf->erase(conv_data[CONV_SPL]->tf->begin()+1, conv_data[CONV_SPL]->tf->end());
        }
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
void Convolution::add_accepted_tf(int c_acc_it_min, int c_acc_it_max){
    Gauss g_x    , g_y    , g_angle,
          g_x_inv, g_y_inv, g_angle_inv;
    Gauss gt_x    , gt_y    , gt_angcos    , gt_angsin,
          gt_x_inv, gt_y_inv, gt_angcos_inv, gt_angsin_inv;
    cv::Matx33d T;
    cv::Matx33d Ti;
    xy com     = conv_data[CONV_REF]->com;
    xy com_inv = conv_data[CONV_SPL]->com;
    double weight_sum = 0;
    double score = 0;

    for(int i = c_acc_it_min; i < c_acc_it_max; i++){
        T  = conv_accepted[i]->T;
        xy tf_com     = mat_mult(T , com    );

        Ti = T.inv();
        xy tf_com_inv = mat_mult(Ti, com_inv);

        double weight  = conv_accepted[i]->score;

        double cos_angle;

        g_x          .add_w_sample((tf_com     - com).x, weight);
        g_y          .add_w_sample((tf_com     - com).y, weight);
        cos_angle = T(0,0);
        if     (cos_angle >   1.0){ cos_angle =   1.0;}
        else if(cos_angle < - 1.0){ cos_angle = - 1.0;}
        g_angle      .add_w_sample(acos(cos_angle)       , weight);
        gt_x         .add_w_sample(T(2) ,weight);
        gt_y         .add_w_sample(T(5) ,weight);
        gt_angcos    .add_w_sample(T(0) ,weight);
        gt_angsin    .add_w_sample(T(3) ,weight);

        g_x_inv      .add_w_sample((tf_com_inv - com_inv).x, weight);
        g_y_inv      .add_w_sample((tf_com_inv - com_inv).y, weight);
        cos_angle = Ti(0,0);
        if     (cos_angle >   1.0){ cos_angle =   1.0;}
        else if(cos_angle < - 1.0){ cos_angle = - 1.0;}
        g_angle_inv  .add_w_sample(acos(cos_angle)           , weight);
        gt_x_inv     .add_w_sample(Ti(2),weight);
        gt_y_inv     .add_w_sample(Ti(5),weight);
        gt_angcos_inv.add_w_sample(Ti(0),weight);
        gt_angsin_inv.add_w_sample(Ti(3),weight);

        weight_sum += weight;
        score      += sqr(weight);
    }
    score /= weight_sum;

    double cov_xy = 0, cov_xy_inv = 0;
    for(int i = c_acc_it_min; i < c_acc_it_max; i++){//TODO covariance computation is not perfect
        T  = conv_accepted[i]->T;
        xy tf_com     = mat_mult(T , com    );

        Ti = T.inv();
        xy tf_com_inv = mat_mult(Ti, com_inv);

        double weight  = conv_accepted[i]->score;

        double a     = ( (tf_com     - com    ).x - g_x.getMean()     ) ;
        double b     = ( (tf_com     - com    ).y - g_y.getMean()     ) ;
        cov_xy       += a     * b     * weight / weight_sum;

        double a_inv = ( (tf_com_inv - com_inv).x - g_x_inv.getMean() ) ;
        double b_inv = ( (tf_com_inv - com_inv).y - g_y_inv.getMean() ) ;
        cov_xy_inv   += a_inv * b_inv * weight / weight_sum;
    }
    T  = cv::Matx33d(gt_angcos    .getMean(), -gt_angsin    .getMean(), gt_x    .getMean(),
                     gt_angsin    .getMean(),  gt_angcos    .getMean(), gt_y    .getMean(),
                                           0,                        0,                 1);
    Ti = cv::Matx33d(gt_angcos_inv.getMean(), -gt_angsin_inv.getMean(), gt_x_inv.getMean(),
                     gt_angsin_inv.getMean(),  gt_angcos_inv.getMean(), gt_y_inv.getMean(),
                                           0,                        0,                 1);
    cv::Matx22d C (g_x.getVariance()    , cov_xy              ,
                   cov_xy               , g_y.getVariance()    );
    cv::Matx22d Ci(g_x_inv.getVariance(), cov_xy_inv          ,
                   cov_xy_inv           , g_y_inv.getVariance());


    //HERE ADD TF_COM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    TfVar tf_var    (com    , T , xy(g_x    .getMean(), g_y    .getMean()), C , atan2(gt_angsin    .getMean(), gt_angcos    .getMean()), g_angle    .getVariance());
    TfVar tf_var_inv(com_inv, Ti, xy(g_x_inv.getMean(), g_y_inv.getMean()), Ci, atan2(gt_angsin_inv.getMean(), gt_angcos_inv.getMean()), g_angle_inv.getVariance());

    conv_data[CONV_REF]->tf->push_back(TFdata(conv_accepted[0]->seg_spl, CONV_REF, score, tf_var    , tf_var_inv));
    conv_data[CONV_SPL]->tf->push_back(TFdata(conv_accepted[0]->seg_ref, CONV_SPL, score, tf_var_inv, tf_var    ));
}
//void Convolution::add_accepted_tf(int c_acc_it_min, int c_acc_it_max){//maybe try with mean of the tf
//    Gauss g_x    , g_y    , g_angle,
//          g_x_inv, g_y_inv, g_angle_inv;
//    Gauss gt_x    , gt_y    , gt_angcos    , gt_angsin,
//          gt_x_inv, gt_y_inv, gt_angcos_inv, gt_angsin_inv;
//    cv::Matx33d T;
//    cv::Matx33d Ti;
//    xy com     = conv_data[CONV_REF]->com;
//    xy com_inv = conv_data[CONV_SPL]->com;
//    double weight_sum = 0;
//    double score = 0;

//    for(int i = c_acc_it_min; i < c_acc_it_max; i++){
//        T  = conv_accepted[i]->T;
//        xy tf_com     = mat_mult(T , com    );

//        Ti = T.inv();
//        xy tf_com_inv = mat_mult(Ti, com_inv);

//        double weight  = conv_accepted[i]->score;

//        double cos_angle;

//        g_x          .add_w_sample((tf_com     - com).x, weight);
//        g_y          .add_w_sample((tf_com     - com).y, weight);
//        cos_angle = T(0,0);
//        if     (cos_angle >   1.0){ cos_angle =   1.0;}
//        else if(cos_angle < - 1.0){ cos_angle = - 1.0;}
//        g_angle      .add_w_sample(acos(cos_angle)       , weight);
//        gt_x         .add_w_sample(T(2) ,weight);
//        gt_y         .add_w_sample(T(5) ,weight);
//        gt_angcos    .add_w_sample(T(0) ,weight);
//        gt_angsin    .add_w_sample(T(3) ,weight);

//        g_x_inv      .add_w_sample((tf_com_inv - com_inv).x, weight);
//        g_y_inv      .add_w_sample((tf_com_inv - com_inv).y, weight);
//        cos_angle = Ti(0,0);
//        if     (cos_angle >   1.0){ cos_angle =   1.0;}
//        else if(cos_angle < - 1.0){ cos_angle = - 1.0;}
//        g_angle_inv  .add_w_sample(acos(cos_angle)           , weight);
//        gt_x_inv     .add_w_sample(Ti(2),weight);
//        gt_y_inv     .add_w_sample(Ti(5),weight);
//        gt_angcos_inv.add_w_sample(Ti(0),weight);
//        gt_angsin_inv.add_w_sample(Ti(3),weight);

//        weight_sum += weight;
//        score      += sqr(weight);
//    }
//    score /= weight_sum;

//    double cov_xy = 0, cov_xy_inv = 0;
//    for(int i = c_acc_it_min; i < c_acc_it_max; i++){//TODO covariance computation is not perfect
//        T  = conv_accepted[i]->T;
//        xy tf_com     = mat_mult(T , com    );

//        Ti = T.inv();
//        xy tf_com_inv = mat_mult(Ti, com_inv);

//        double weight  = conv_accepted[i]->score;

//        double a     = ( (tf_com     - com    ).x - g_x.getMean()     ) ;
//        double b     = ( (tf_com     - com    ).y - g_y.getMean()     ) ;
//        cov_xy       += a     * b     * weight / weight_sum;

//        double a_inv = ( (tf_com_inv - com_inv).x - g_x_inv.getMean() ) ;
//        double b_inv = ( (tf_com_inv - com_inv).y - g_y_inv.getMean() ) ;
//        cov_xy_inv   += a_inv * b_inv * weight / weight_sum;
//    }
//    T  = cv::Matx33d(gt_angcos    .getMean(), -gt_angsin    .getMean(), gt_x    .getMean(),
//                     gt_angsin    .getMean(),  gt_angcos    .getMean(), gt_y    .getMean(),
//                                           0,                        0,                 1);
//    Ti = cv::Matx33d(gt_angcos_inv.getMean(), -gt_angsin_inv.getMean(), gt_x_inv.getMean(),
//                     gt_angsin_inv.getMean(),  gt_angcos_inv.getMean(), gt_y_inv.getMean(),
//                                           0,                        0,                 1);
//    cv::Matx22d C (g_x.getVariance()    , cov_xy              ,
//                   cov_xy               , g_y.getVariance()    );
//    cv::Matx22d Ci(g_x_inv.getVariance(), cov_xy_inv          ,
//                   cov_xy_inv           , g_y_inv.getVariance());


//    //HERE ADD TF_COM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//    TfVar tf_var    (com    , T , xy(g_x    .getMean(), g_y    .getMean()), C , atan2(gt_angsin    .getMean(), gt_angcos    .getMean()), g_angle    .getVariance());
//    TfVar tf_var_inv(com_inv, Ti, xy(g_x_inv.getMean(), g_y_inv.getMean()), Ci, atan2(gt_angsin_inv.getMean(), gt_angcos_inv.getMean()), g_angle_inv.getVariance());

//    conv_data[CONV_REF]->tf->push_back(TFdata(conv_accepted[0]->seg_spl, CONV_REF, score, tf_var    , tf_var_inv));
//    conv_data[CONV_SPL]->tf->push_back(TFdata(conv_accepted[0]->seg_ref, CONV_SPL, score, tf_var_inv, tf_var    ));
//}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::sample_const_dist(SegmentDataExtPtr &input, ConvolStatus conv_stat){
    if(input->conv){//if segment was sampled, dont do anything
        conv_data[conv_stat] = input->conv;
        return;
    }
    conv_data[conv_stat] = boost::shared_ptr<ConvData>(new ConvData(input));
    double rest=0;
    xy p_now,p1_last,p2_last;

    xy com;
    for(PointDataVectorIter it_in=input->p.begin();it_in!=--input->p.end();it_in++){
        p1_last=to_xy(*it_in);
        p2_last=to_xy(*++it_in);
        while(rest+diff(to_polar(p1_last),*it_in) > sample_dist){

            double k=(sample_dist-rest)/(diff(to_polar(p1_last),*it_in));
            p_now=xy(p1_last.x+(p2_last.x-p1_last.x)*k,p1_last.y+(p2_last.y-p1_last.y)*k);

            p1_last=p_now;

            ///TODO: do a more inteligent way of calculating normals
            double alfa_normal;//normal computation
            --it_in;
            alfa_normal=atan2(p2_last.y-to_xy(*it_in).y,p2_last.x-to_xy(*it_in).x)+M_PI/2.0;
            ++it_in;

            com += p_now;//com computation

            conv_data[conv_stat]->p_cd->push_back(PointDataSample(to_polar(p_now),NULL,alfa_normal));

            rest=0;
        }
        rest+=diff(to_polar(p1_last),*it_in);
        --it_in;
    }
    smooth_normals(conv_stat);//smooth the normals

    conv_data[conv_stat]->com = xy(com.x / conv_data[conv_stat]->p_cd->size(), com.y / conv_data[conv_stat]->p_cd->size());
    input->conv = conv_data[conv_stat];
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void Convolution::smooth_normals(ConvolStatus conv_stat){//smooths normals, erases unsmoothable ones and creates aproximated angle classification database
    boost::shared_ptr<ConvData> temp( new ConvData(conv_data[conv_stat]->seg) );

    for(int i=0;i<conv_data[conv_stat]->p_cd->size();i++){
        int len = (smooth_mask.size()-1)/2;

        int not_included = std::max(0,i-len)-( i-len );
        not_included = std::max(not_included, -std::min((int)(conv_data[conv_stat]->p_cd->size())-1,i+len)+(len+i));
        if(not_included > 0){
//            continue;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!here which is not included, non smoothable, maybe better to accept everything but add a low-weight stuff
        }

        len = len - not_included;

        double val = 0;
        double w_sum = 0;
        std::list<double>::iterator smooth_mask_it = smooth_mask.begin();
        while(not_included--){ smooth_mask_it++; }

        for(int j=i-len;j<=i+len;j++){
            double angg = conv_data[conv_stat]->p_cd->at(j).normal_ang - conv_data[conv_stat]->p_cd->at(i).normal_ang;
            normalizeAngle(angg);
            val+= angg * *smooth_mask_it;
            w_sum+=*smooth_mask_it;
            smooth_mask_it++;
        }

        if(w_sum != 0){
            val/=w_sum;
        }
        val += conv_data[conv_stat]->p_cd->at(i).normal_ang;
        temp->p_cd->push_back(conv_data[conv_stat]->p_cd->at(i));
        temp->p_cd->back().normal_ang = val;
    }
    conv_data[conv_stat] = temp;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
///------------------------------------------------------------------------------------------------------------------------------------------------///

void AngKeyDataBase::init(){
    keys.resize((int)(2.0 * M_PI / d_angl));
    no_keys = 0;
    for( std::vector<int>::iterator it = keys.begin(); it != keys.end(); it++ ){
        *it = 0;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

int* AngKeyDataBase::add(double angle){

    int ang_key = normalize_key((int)(round(angle / d_angl) + keys.size() / 2.0));
    if(keys.at(ang_key) == 0){
        no_keys++;
    }
    keys.at(ang_key) += 1.0;
    return &keys.at(ang_key);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

int* AngKeyDataBase::remove(double angle){
    int ang_key = normalize_key((int)(round(angle / d_angl) + keys.size() / 2.0));
    keys.at(ang_key) -= 1.0;
    if(keys.at(ang_key) == 0){
        no_keys--;
    }
    return &keys.at(ang_key);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

int AngKeyDataBase::normalize_key(int ang_key){
    while(ang_key > (int)keys.size() - 1){
      ang_key -= keys.size();
    }
    while(ang_key < 0){
      ang_key += keys.size();
    }
    return ang_key;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
