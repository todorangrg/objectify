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
    dir_followed_fin(UNK){}


///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::run(double _pred_time){

    SegmentDataPtrVectorPtr    seg_init   ; SegCopy(k.seg_init, seg_init);

    if(k.pos_init){
        RState rob_now(k.S); KInp u; u.v = k.S.at<double>(3); u.w = k.S.at<double>(4); u.dt = 0.5; RState d_rob = k.predict_rob_pos(rob_now, u);
        RState rob_fut;  rob_fut.xx = rob_now.xx + d_rob.xx; rob_fut.xy = rob_now.xy + d_rob.xy; rob_fut.xphi = rob_now.xphi + d_rob.xphi;
        robot_now = rob_now; robot_fut = rob_fut;
    }
    else{ robot_now = RState(0,0,0); robot_fut = RState(0,0,0); }
    n2f.init(robot_now, robot_fut);

    //here you should bloat the init image
    segmentation.run_future(seg_init, seg_ext_now, seg_ext_ftr, n2f);

    double        d_followed_old =   d_followed_fin;
    if(k.Oi.count(o_followed_fin) == 0){ o_followed_fin.reset(); }
    ObjectDataPtr o_followed_old =   o_followed_fin;
    int         dir_followed_old = dir_followed_fin;
    double        d_followed;
    ObjectDataPtr o_followed;
    int         dir_followed;

    full_potential = xy(0,0);

    w2r.init(RState(0,0,0), robot_now);
    potential_weight(seg_ext_now, k.goal, w2r, NOW, full_potential);
    tangent_bug     (seg_ext_now, k.goal, w2r, NOW, robot_now, d_followed_old, o_followed_old, d_followed, o_followed);
    find_dir_followed(NOW, dir_followed_old, o_followed_old, dir_followed, o_followed, d_followed, target);

//    if(k.pos_init){
//        d_followed_old = d_followed; o_followed_old = o_followed; dir_followed_old = dir_followed;
//        w2r.init(RState(0,0,0), robot_fut);
//        potential_weight(seg_ext_ftr, k.goal, w2r, FTR);
//        tangent_bug     (seg_ext_now, k.goal, w2r, NOW, d_followed_old, o_followed_old, d_followed, o_followed);

//        dir_followed = find_dir_followed(dir_followed_old, o_followed_old, o_followed, d_followed);
//        //here if something new found you have to tf back in the NOW frame
//    }

    d_followed_fin   = d_followed;
    o_followed_fin   = o_followed;
    dir_followed_fin = dir_followed;



    vel_controller(full_potential, target);
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::vel_controller(xy & full_pot, polar target){
    full_pot *= 0.001;
    polar controller_target = to_polar(full_pot + to_xy(target));//pot_scale

    //target
    cmd_vel.w = 0.5  * controller_target.angle;                         //w_kp_goal,
    cmd_vel.v = fmin(0.01 / fabs(cmd_vel.w), 0.2 * controller_target.r);//v_kp_goal, v_kp_w,

    double v_max = 0.5, w_max = 0.4;
    if(abs(cmd_vel.v)>v_max)  cmd_vel.v=sgn(cmd_vel.v)*v_max;//v_max, w_max
    if(abs(cmd_vel.w)>w_max)  cmd_vel.w=sgn(cmd_vel.w)*w_max;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void TangentBug::find_dir_followed(int frame, int   dir_followed_old, ObjectDataPtr o_followed_old,
                                              int & dir_followed    , ObjectDataPtr o_followed    , double d_followed, polar & target){
    polar goal_local = to_polar(w2r.ro2rn(k.goal)); goal_local.r = fmin(sensor_range_max, goal_local.r);

    if(d_followed == -1){ target = goal_local; dir_followed = UNK; }//if free direction
    else{
        if(o_followed == o_followed_old){ dir_followed = dir_followed_old; }//if following the same obstacle propagate wall direction
        else                            { dir_followed = UNK;              }

        if(dir_followed == UNK){//if wall direction unknown find the closest to goal
            if(diff(to_xy(goal_local),to_xy(o_followed->ang_bounds[NOW][NEG])) <
               diff(to_xy(goal_local),to_xy(o_followed->ang_bounds[NOW][POS]))){ dir_followed = NEG; }
            else{                                                                dir_followed = POS; }
        }
        target = o_followed->ang_bounds[NOW][dir_followed];
    }
    //here if something new found you have to tf back in the NOW frame
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>
void TangentBug::potential_weight(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_w2r, int frame, xy & full_pot){
    for(std::map<ObjectDataPtr, ObjMat>::iterator oi = k.Oi.begin(); oi != k.Oi.end(); oi++){
        oi->first->wall_potential[frame] = xy(0,0);
    }
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

            if(pp == (*ss)->p.begin()){ ang_bounds[NEG] = p; ang_bounds[POS] = p; }
            else{
                double ang_diff = p.angle - ang_bounds[NEG].angle;
                if(/*normalizeAngle(*/ang_diff/*)*/ < 0){
                    ang_bounds[NEG] = p;
                }
                      ang_diff = p.angle - ang_bounds[POS].angle;
                if(/*normalizeAngle(*/ang_diff/*)*/ > 0){
                    ang_bounds[POS] = p;
                }
            }

            double dist_to_goal = sqrt(sqr(to_xy(p).x - tf_w2r.ro2rn(goal).x) + sqr(to_xy(p).y - tf_w2r.ro2rn(goal).y));
            if( dist_to_goal < dist_to_goal_min ){ dist_to_goal_min = dist_to_goal; }

            p.r = sqr(1.0 / p.r);
            pot -= to_xy(p);
        }
        (*ss)->getObj()->closest_d     [frame]      = d_to_rob_min;
        (*ss)->getObj()->wall_potential[frame]     += pot;
        (*ss)->getObj()->dist_to_goal  [frame]      = dist_to_goal_min;
        (*ss)->getObj()->ang_bounds    [frame][NEG] = ang_bounds[NEG];
        (*ss)->getObj()->ang_bounds    [frame][POS] = ang_bounds[POS];

        full_pot += pot;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template <class SegData>//for free_space => d_followed = -1;
void TangentBug::tangent_bug(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, xy goal, FrameTf tf_w2r, int frame, RState rob_pos,
                                 double d_followed_old, ObjectDataPtr & o_followed_old, double & d_followed_new, ObjectDataPtr & o_followed_new){

    double bloat_size = 0.2;
    polar goal_local = to_polar(tf_w2r.ro2rn(goal));
    Line  goal_line  = get_line_param(xy(0,0),to_xy(goal_local));
    bool free_space = true;
    double angle_to_min = goal_local.angle - angle_min;
    double angle_to_max = goal_local.angle - angle_max;

    goal_local.r = goal_local.r + bloat_size;
    if(!(((/*normalizeAngle(*/angle_to_min/*)*/ < 0)||(/*normalizeAngle(*/angle_to_max/*)*/ > 0))&&(d_followed_old != -1))){
        for(std::map<ObjectDataPtr, ObjMat>::iterator oi = k.Oi.begin(); oi != k.Oi.end(); oi++){
            double ang_diff[2];
            ang_diff[NEG] = goal_local.angle - oi->first->ang_bounds[frame][NEG].angle ;
            ang_diff[POS] = goal_local.angle - oi->first->ang_bounds[frame][POS].angle ;
            if((/*normalizeAngle(*/ang_diff[NEG]/*)*/ > 0)&&(/*normalizeAngle(*/ang_diff[POS]/*)*/ < 0)){//if found an ang bound
                free_space  = false;
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
                if(inside){ free_space = true; break; }
            }
        }
//        for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != data->end(); ss++){
//            double ang_diff[2];
//            ang_diff[NEG] = goal_local.angle - (*ss)->getObj()->ang_bounds[frame][NEG].angle ;
//            ang_diff[POS] = goal_local.angle - (*ss)->getObj()->ang_bounds[frame][POS].angle ;
//            if((/*normalizeAngle(*/ang_diff[NEG]/*)*/ > 0)&&(/*normalizeAngle(*/ang_diff[POS]/*)*/ < 0)){//if found an ang bound
//                //possible free_space = false; have to check if target is closer than object border
//                free_space  = false;
//                bool inside = true;
//                for(PointDataVectorIter pp = (*ss)->p.begin(); pp != (*ss)->p.end(); pp++){
//                    polar p = to_polar(tf_sns.s2r(to_xy(*pp)));
//                    xy p_inters_line;
//                    double line_dist = get_dist_p(goal_line, to_xy(p), &p_inters_line);
//                    if(((p_inters_line.x > fmin(xy(0,0).x,to_xy(goal_local).x))&&(p_inters_line.x < fmax(xy(0,0).x,to_xy(goal_local).x))&&
//                        (p_inters_line.y > fmin(xy(0,0).y,to_xy(goal_local).y))&&(p_inters_line.y < fmax(xy(0,0).y,to_xy(goal_local).y))&&
//                        (line_dist < bloat_size))){
//                        inside = false; break;
//                    }
//                }
//                if(inside){ free_space = true; break; }
//            }
//        }
    } else { free_space = false; }
    goal_local.r = goal_local.r - bloat_size;

    //if target in free space or within an object //free going
    if(free_space){ d_followed_new = -1; return; }

    if(d_followed_old == -1){ d_followed_new = diff(goal, xy(rob_pos.xx, rob_pos.xy)); }
    else                    { d_followed_new = d_followed_old ; }
    o_followed_new = o_followed_old;
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator ss = data->begin(); ss != data->end(); ss++){
        if((*ss)->getObj()->dist_to_goal[frame] < d_followed_new){
            d_followed_new = (*ss)->getObj()->dist_to_goal[frame];
            o_followed_new = (*ss)->getObj();
        }
    }
    return;
}


////between the 2 measurements, choose the newest wall if found diferent; if the same, use the d_min shizzle


////!!!!!!!!!!!!!!PROPAGATE LAST POINT ACCORDING TO THE TF OF THE OBJECT THAT HAS SMALLEST VELOCITY IN YOUR FIELD OF VIEW (IF ALL BIG, KEEP IN STATIC)
