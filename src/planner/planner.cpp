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


#include "planner/planner.h"

using namespace cv;

enum{
    NOW,
    FTR
};
enum{
    NEG,
    POS,
    UNK
};

///------------------------------------------------------------------------------------------------------------------------------------------------///

TangentBug::TangentBug(RecfgParam& param, SensorTf& _tf_sns, KalmanSLDM & _k, Segmentation &_segmentation) :
    sensor_range_max(param.sensor_r_max),
    angle_max(param.cb_sensor_point_angl_max),
    angle_min(param.cb_sensor_point_angl_min),
    tf_sns(_tf_sns),
    k(_k),
    segmentation(_segmentation),
    d_followed_fin(-1),
    dir_followed_fin(UNK),
    pot_scale(param.planner_pot_scale),
    w_kp_goal(param.planner_w_kp_goal),
    w_kd_goal(param.planner_w_kd_goal),
    v_kp_w(param.planner_v_kp_w),
    v_kp_goal(param.planner_v_kp_goal),
    v_kd_goal(param.planner_v_kd_goal),
    v_max(param.planner_v_max),
    w_max(param.planner_w_max),
    d_to_obj(0){}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::run(double _pred_time, KInp u){

    SegmentDataPtrVectorPtr    seg_init   ; SegCopy(k.seg_init, seg_init);

    if(k.pos_init){
        RState rob_now(k.S); u.v = k.S.at<double>(3); u.w = k.S.at<double>(4);
        u.dt = fmin(1.0, fmax(0.1, d_to_obj));
        RState d_rob = k.predict_rob_pos(rob_now, u.dt);
        RState rob_fut;  rob_fut.xx = rob_now.xx + d_rob.xx; rob_fut.xy = rob_now.xy + d_rob.xy; rob_fut.xphi = rob_now.xphi + d_rob.xphi;
        robot_now = rob_now; robot_fut = rob_fut;


    }
    else{ robot_now = RState(0,0,0); robot_fut = RState(0,0,0); }
    n2f.init(robot_now, robot_fut);

    //here you should bloat the init image
    segmentation.run_future(seg_init, seg_ext_now, seg_ext_ftr, n2f, robot_fut, u, k);

    if(k.Oi.count(o_followed_fin) == 0){ o_followed_fin.reset(); d_followed_fin = -1;}
    double        d_followed_old =   d_followed_fin;
    ObjectDataPtr o_followed_old =   o_followed_fin;
    int         dir_followed_old = dir_followed_fin;
    double        d_followed;
    ObjectDataPtr o_followed;
    int         dir_followed;

    full_potential = xy(0,0);

    w2r.init(RState(0,0,0), robot_now);
//    if(!k.pos_init){
        potential_weight(seg_ext_now, k.goal, w2r, NOW, full_potential);
//        tangent_bug     (seg_ext_now, k.goal, w2r, NOW, robot_now, d_followed_old, o_followed_old, d_followed, o_followed);
//        find_dir_followed(seg_ext_now, NOW, dir_followed_old, o_followed_old, dir_followed, o_followed, d_followed, target);

//        d_followed_fin   = d_followed;
//        o_followed_fin   = o_followed;
//        dir_followed_fin = dir_followed;
//    }

    double        d_followed_ftr;
    ObjectDataPtr o_followed_ftr;
    int         dir_followed_ftr;
    if(k.pos_init){
//        d_followed_old = d_followed; o_followed_old = o_followed; dir_followed_old = dir_followed;
        w2r.init(RState(0,0,0), robot_now);
        potential_weight(seg_ext_ftr, k.goal, w2r, FTR, full_potential);
        tangent_bug     (seg_ext_ftr, k.goal, w2r, FTR, robot_now, d_followed_old, o_followed_old, d_followed_ftr, o_followed_ftr);

        find_dir_followed(seg_ext_ftr, FTR, dir_followed_old, o_followed_old, dir_followed_ftr, o_followed_ftr, d_followed_ftr, target);

        if((o_followed)||(o_followed_ftr)){
//            if(o_followed == o_followed_ftr){
                polar goal_local = to_polar(w2r.ro2rn(k.goal)); goal_local.r = fmin(sensor_range_max, goal_local.r);
                double d_r2o[2][2];
                d_r2o[NOW][NEG] = o_followed_ftr->ang_bounds[NOW][NEG].r;
                d_r2o[NOW][POS] = o_followed_ftr->ang_bounds[NOW][POS].r;
                d_r2o[FTR][NEG] = o_followed_ftr->ang_bounds[FTR][NEG].r;
                d_r2o[FTR][POS] = o_followed_ftr->ang_bounds[FTR][POS].r;
                double d_o2g[2][2];
                d_o2g[NOW][NEG] = diff(o_followed_ftr->ang_bounds[NOW][NEG], goal_local);
                d_o2g[NOW][POS] = diff(o_followed_ftr->ang_bounds[NOW][POS], goal_local);
                d_o2g[FTR][NEG] = diff(o_followed_ftr->ang_bounds[FTR][NEG], goal_local);
                d_o2g[FTR][POS] = diff(o_followed_ftr->ang_bounds[FTR][POS], goal_local);

                double d_alf_r2o[2];
                d_alf_r2o[NEG]  = o_followed_ftr->ang_bounds[FTR][NEG].angle/* - o_followed->ang_bounds[NOW][NEG].angle*/;
                d_alf_r2o[POS]  = o_followed_ftr->ang_bounds[FTR][POS].angle/* - o_followed->ang_bounds[NOW][POS].angle*/;
                normalizeAngle(d_alf_r2o[NEG]);
                normalizeAngle(d_alf_r2o[POS]);

                if(((d_r2o[FTR][ dir_followed_ftr] + d_o2g[FTR][ dir_followed_ftr]) - (d_r2o[NOW][ dir_followed_ftr] + d_o2g[NOW][ dir_followed_ftr])) /** fabs(sin(fabs(d_alf_r2o[ dir_followed_ftr])))*/ >
                   ((d_r2o[FTR][!dir_followed_ftr] + d_o2g[FTR][!dir_followed_ftr]) - (d_r2o[NOW][!dir_followed_ftr] + d_o2g[NOW][!dir_followed_ftr])) /** fabs(sin(fabs(d_alf_r2o[!dir_followed_ftr])))*/){
                    target = o_followed_ftr->ang_bounds[FTR][!dir_followed_ftr];
                    dir_followed_ftr = !dir_followed_ftr;
                }
                else{
                    target = o_followed_ftr->ang_bounds[FTR][dir_followed_ftr];
                }
//            }
        }
        //dir_followed = find_dir_followed(dir_followed_old, o_followed_old, o_followed, d_followed);
        //here if something new found you have to tf back in the NOW frame
//        if(o_followed_ftr){
            d_followed_fin   = d_followed_ftr;
            o_followed_fin   = o_followed_ftr;
            dir_followed_fin = dir_followed_ftr;
//        }
    }


    if(o_followed_fin){
        if(u.v == 0){ u.v == 0.0001; }
        d_to_obj = fmax(o_followed_fin->closest_d[FTR] / fabs(u.v), 0.1);
    }
    else{
        d_to_obj = fmin(1.0, d_to_obj + 0.1);//100;
    }

    if(!k.pos_init){
        vel_controller(seg_ext_now, full_potential, target, u);
    }
    else{
        vel_controller(seg_ext_ftr, full_potential, target, u);
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::vel_controller(SegmentDataExtPtrVectorPtr &data, xy & full_pot, polar target, KInp u){
    Line  goal_line  = get_line_param(xy(0,0),to_xy(target));
    polar p_follow_base = target;
    polar p_follow_ext  = target;
    for(SegmentDataExtPtrVectorIter ss = data->begin(); ss != data->end(); ss++){
        for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
            polar p = to_polar(tf_sns.s2r(to_xy(*pp)));
            xy p_inters_line;
            double line_dist = get_dist_p(goal_line, to_xy(p), &p_inters_line);
            if(((p_inters_line.x > fmin(xy(0,0).x,to_xy(target).x))&&(p_inters_line.x < fmax(xy(0,0).x,to_xy(target).x))&&
                (p_inters_line.y > fmin(xy(0,0).y,to_xy(target).y))&&(p_inters_line.y < fmax(xy(0,0).y,to_xy(target).y))&&
                (line_dist < 0.4)&&(line_dist > 0.01))){//hardcoded
//                if(to_polar(p_inters_line).r < p_follow_base.r){
                    polar  p_follow_base_now = to_polar(p_inters_line);
                    double d_remaining = 0.4 - line_dist;
                    double d_angle = p.angle - target.angle; if(normalizeAngle(d_angle) > 0){ d_remaining = - d_remaining; }
                    polar  p_follow_ext_now = polar(sqrt(sqr(p_follow_base_now.r) + sqr(d_remaining)), p_follow_base_now.angle + atan2(d_remaining, p_follow_base_now.r));
                    double test_angle1 = p_follow_ext_now.angle - target.angle;
                    double test_angle2 = p_follow_ext.angle - target.angle;
                    if(fabs(normalizeAngle(test_angle1)) > fabs(normalizeAngle(test_angle2))){
                        p_follow_ext = p_follow_ext_now;
                    }
//                }
            }
        }

    }
    normalizeAngle(p_follow_ext.angle);
    target_to_follow = p_follow_ext;

    full_pot *= pot_scale;
    polar full_pot_polar = to_polar(full_pot);
    full_pot_polar.r = full_pot_polar.r * cos(full_pot_polar.angle - target.angle) * sgn(- sin(full_pot_polar.angle - target.angle));
    full_pot_polar.angle = target.angle + M_PI / 2.0;
    full_pot = to_xy(full_pot_polar);
    polar controller_target = to_polar(to_xy(target_to_follow)/*full_pot + to_xy(target)*/);//pot_scale
    controller_target.r = target_to_follow.r + diff(target_to_follow, target);

    static polar target_old = polar(0,0);

    if(target.r < 0.1){
        cmd_vel.w = 0;
        cmd_vel.v = 0;
        target_old = polar(0,0);
        return;
    }

    //target
    cmd_vel.w = w_kp_goal  * controller_target.angle + w_kd_goal * (controller_target.angle - target_old.angle) / u.dt;
    cmd_vel.v = fmax(0.0,fmin(v_kp_w * 0.5/fabs(controller_target.angle),
                              v_kp_goal * controller_target.r + v_kd_goal *(controller_target.r - target_old.r) / u.dt));


    target_old = controller_target/*target*/;

    //double v_max = 0.5, w_max = 0.4;
    if(abs(cmd_vel.v)>v_max)  cmd_vel.v=sgn(cmd_vel.v)*v_max;//v_max, w_max
    if(abs(cmd_vel.w)>w_max)  cmd_vel.w=sgn(cmd_vel.w)*w_max;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::find_dir_followed(SegmentDataExtPtrVectorPtr &data, int frame, int   dir_followed_old, ObjectDataPtr o_followed_old,
                                                                            int & dir_followed    , ObjectDataPtr & o_followed    , double d_followed, polar & target){
    polar goal_local = to_polar(w2r.ro2rn(k.goal)); goal_local.r = fmin(sensor_range_max, goal_local.r);

    if(d_followed == -1){ target = goal_local; dir_followed = UNK; }//if free direction
    else{
        bool same_obj = false;
//        if(!o_followed){ return; }
        for(int i=0; i < o_followed->parrents_merge.size(); i++){
            if(o_followed->parrents_merge[i] == o_followed_old){
                same_obj = true; break;
            }
        }
        if((o_followed == o_followed_old)||(same_obj)){ dir_followed = dir_followed_old; }//if following the same obstacle propagate wall direction
        else                                          { dir_followed = UNK;              }

        if(dir_followed == UNK){//if wall direction unknown find the closest to goal
            if(diff(to_xy(goal_local),to_xy(o_followed->ang_bounds[frame][NEG])) <
               diff(to_xy(goal_local),to_xy(o_followed->ang_bounds[frame][POS]))){ dir_followed = NEG; }
            else{                                                                  dir_followed = POS; }
        }
        if(o_followed->ang_b_valid[frame][dir_followed] == false){
            SegmentDataExtPtrVectorIter ss;
            for(ss = data->begin(); ss != data->end(); ss++){
                if((*ss)->getObj() == o_followed){ break; }
            }
            if(dir_followed == NEG){
                while(((*ss)->getObj()->ang_b_valid[frame][dir_followed] == false)&&(ss != data->begin())){
                    ss--;
                }
            }
            if(dir_followed == POS){
                while(((*ss)->getObj()->ang_b_valid[frame][dir_followed] == false)&&(ss != --data->end())){
                    ss++;
                }
            }
            target = (*ss)->getObj()->ang_bounds[frame][dir_followed];
            o_followed = (*ss)->getObj();
            return;
        }

        target = o_followed->ang_bounds[frame][dir_followed];
    }
    //here if something new found you have to tf back in the NOW frame
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void TangentBug::potential_weight(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_w2r, int frame, xy & full_pot){
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = k.Oi.begin(); oi != k.Oi.end(); oi++){
        oi->first->wall_potential[frame] = xy(0,0);
        oi->first->ang_bounds[frame][NEG].r = -1000;
        oi->first->ang_bounds[frame][POS].r = -1000;
    }
    int p_no = 0;
    double angle_shift_neg = 0;
    double angle_shift_pos = 0;
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != data->end(); ss++){
        xy pot(0,0);
        double dist_to_goal_min = (*ss)->getObj()->dist_to_goal[frame];
        polar  ang_bounds[2];
        double d_to_rob_min = sensor_range_max;
        for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
            polar p = to_polar(tf_sns.s2r(to_xy(*pp)));

            if(diff(to_xy(p), xy(0,0)) < d_to_rob_min){
                d_to_rob_min = diff(to_xy(p), xy(0,0));
            }

            if(pp == (*ss)->p.begin()){
                ang_bounds[NEG] = p; ang_bounds[POS] = p;
                angle_shift_neg = - (M_PI + p.angle);
                angle_shift_pos = - (M_PI + p.angle);
            }
            else{
                double ang_diff = p.angle + angle_shift_neg;
                ang_diff        = M_PI + normalizeAngle(ang_diff);
                if(ang_diff < 0){ ang_bounds[NEG] = p; angle_shift_neg = - (M_PI + p.angle); }
                ang_diff        = p.angle + angle_shift_pos;
                ang_diff        = M_PI + normalizeAngle(ang_diff);
                if(ang_diff > 0){ ang_bounds[POS] = p; angle_shift_pos = - (M_PI + p.angle); }
            }

            double dist_to_goal = sqrt(sqr(to_xy(p).x - tf_w2r.ro2rn(goal).x) + sqr(to_xy(p).y - tf_w2r.ro2rn(goal).y));
            if( dist_to_goal < dist_to_goal_min ){ dist_to_goal_min = dist_to_goal; }

            p.r = sqr(0.2 / p.r);//HARDCODED
            pot -= to_xy(p);
            p_no++;
        }
        (*ss)->getObj()->closest_d     [frame]      = d_to_rob_min;
        (*ss)->getObj()->wall_potential[frame]     += pot;
        (*ss)->getObj()->dist_to_goal  [frame]      = dist_to_goal_min;

        angle_shift_neg = - (M_PI + (*ss)->getObj()->ang_bounds[frame][NEG].angle);
        double ang_diff = ang_bounds[NEG].angle + angle_shift_neg;
        ang_diff        = M_PI + normalizeAngle(ang_diff);
        if((ang_diff < 0)||((*ss)->getObj()->ang_bounds[frame][NEG].r == -1000)){
            (*ss)->getObj()->ang_bounds    [frame][NEG] = ang_bounds[NEG];
        }
        angle_shift_pos = - (M_PI + (*ss)->getObj()->ang_bounds[frame][POS].angle);
        ang_diff        = ang_bounds[POS].angle + angle_shift_pos;
        ang_diff        = M_PI + normalizeAngle(ang_diff);
        if((ang_diff > 0)||((*ss)->getObj()->ang_bounds[frame][POS].r == -1000)){
            (*ss)->getObj()->ang_bounds    [frame][POS] = ang_bounds[POS];
        }

        full_pot += pot;


        double d_min[2] = {100000, 10000};
        for(typename std::vector<boost::shared_ptr<SegData> >::iterator sss = data->begin(); sss != data->end(); sss++){
            if((*ss)->getObj() == (*sss)->getObj()){continue;}
            for(PointDataVectorIter pp = (*sss)->p.begin(); pp != (*sss)->p.end(); pp++){
                if(d_min[NEG] > diff(ang_bounds[NEG], *pp)){ d_min[NEG] = diff(ang_bounds[NEG], *pp); }
                if(d_min[POS] > diff(ang_bounds[POS], *pp)){ d_min[POS] = diff(ang_bounds[POS], *pp); }
                if((d_min[NEG] < 0.8)&&(d_min[NEG] < 0.8)){ break; }
            }
        }
        if((d_min[NEG] < 0.8)){ (*ss)->getObj()->ang_b_valid[frame][NEG] = false; } else { (*ss)->getObj()->ang_b_valid[frame][NEG] = true; }
        if((d_min[POS] < 0.8)){ (*ss)->getObj()->ang_b_valid[frame][POS] = false; } else { (*ss)->getObj()->ang_b_valid[frame][POS] = true; }
    }
    if(data->size() > 1){
        for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != --data->end(); ss++){
            bool b_neg = (*ss)->getObj()->ang_b_valid[frame][POS]; ss++;
            bool b_pos = (*ss)->getObj()->ang_b_valid[frame][NEG]; ss--;
            if((!b_neg)||(!b_pos)){ (*ss)->getObj()->ang_b_valid[frame][POS] = false; ss++; (*ss)->getObj()->ang_b_valid[frame][NEG] = false; ss--; }
        }
    }

    if(p_no > 0){ full_pot *= 1.0 / (double)p_no; }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>//for free_space => d_followed = -1;
void TangentBug::tangent_bug(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_w2r, int frame, RState rob_pos,
                                 double d_followed_old, ObjectDataPtr & o_followed_old, double & d_followed_new, ObjectDataPtr & o_followed_new){

    double bloat_size = 0.4;//HARDCODED
    polar goal_local = to_polar(tf_w2r.ro2rn(goal));
    Line  goal_line  = get_line_param(xy(0,0),to_xy(goal_local));
    bool free_space = true;
    double angle_to_min = goal_local.angle - angle_min;
    double angle_to_max = goal_local.angle - angle_max;

    goal_local.r = goal_local.r + bloat_size;
    //TODO HERE HERE HERE !!!!!! PROBLEM AGAIN WITH INCLUSION ON THE ANGULAR BOUNDS



    if(!(((/*normalizeAngle(*/angle_to_min/*)*/ < 0)||(/*normalizeAngle(*/angle_to_max/*)*/ > 0))&&(d_followed_old != -1))){
        for(std::map<ObjectDataPtr, ObjMat>::iterator oi = k.Oi.begin(); oi != k.Oi.end(); oi++){
            double ang_diff[2];

            double angle_shift_neg = - (   M_PI + oi->first->ang_bounds[frame][NEG].angle);
            double angle_shift_pos = - ( - M_PI + oi->first->ang_bounds[frame][POS].angle);

            ang_diff[NEG] = goal_local.angle + angle_shift_neg ;
            ang_diff[POS] = goal_local.angle + angle_shift_pos ;

            ang_diff[NEG] =   M_PI + normalizeAngle(ang_diff[NEG]);
            ang_diff[POS] = - M_PI + normalizeAngle(ang_diff[POS]);
            if((/*normalizeAngle(*/ang_diff[NEG]/*)*/ > 0)&&(/*normalizeAngle(*/ang_diff[POS]/*)*/ < 0)){//if found an ang bound
                //free_space  = false;
                bool inside = true;
                for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != data->end(); ss++){
                    if((*ss)->getObj() != oi->first){ continue; }
                    for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
                        polar p = to_polar(tf_sns.s2r(to_xy(*pp)));
                        xy p_inters_line;
                        double line_dist = get_dist_p(goal_line, to_xy(p), &p_inters_line);
                        if(((p_inters_line.x > fmin(xy(0,0).x,to_xy(goal_local).x))&&(p_inters_line.x < fmax(xy(0,0).x,to_xy(goal_local).x))&&
                            (p_inters_line.y > fmin(xy(0,0).y,to_xy(goal_local).y))&&(p_inters_line.y < fmax(xy(0,0).y,to_xy(goal_local).y))&&
                            (line_dist < bloat_size))){
                            inside = false; break;
                        }
                    }

                }
                if(!inside){ free_space = false; break; }
            }
        }
    } else{ free_space = false; }
    goal_local.r = goal_local.r - bloat_size;

    //if target in free space or within an object //free going
    if(free_space){
        d_followed_new = -1;
        return;
    }

    if(d_followed_old == -1){ d_followed_new = 10000/*diff(goal, xy(rob_pos.xx, rob_pos.xy))*/; }
    else                    { d_followed_new = d_followed_old ; }
    o_followed_new = o_followed_old;
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != data->end(); ss++){
        if((*ss)->getObj()->dist_to_goal[frame] < d_followed_new){
            d_followed_new = (*ss)->getObj()->dist_to_goal[frame];
            o_followed_new = (*ss)->getObj();
        }
    }
    if(!o_followed_new){ d_followed_new = -1; }
    return;
}


////between the 2 measurements, choose the newest wall if found diferent; if the same, use the d_min shizzle


////!!!!!!!!!!!!!!PROPAGATE LAST POINT ACCORDING TO THE TF OF THE OBJECT THAT HAS SMALLEST VELOCITY IN YOUR FIELD OF VIEW (IF ALL BIG, KEEP IN STATIC)
