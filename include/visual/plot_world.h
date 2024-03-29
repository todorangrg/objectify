#ifndef PLOT_WORLD_H
#define PLOT_WORLD_H

#include "utils/base_classes.h"
#include "visual/plot.h"

class ConvolInfo;
class KalmanSLDM;

class ObjColor{
public:
    cv::Scalar color;
    int obj_id;
    bool used;
};

class PlotWorld : public Plot{
public:

    void init_w2i();
    void update();
    void plot_points      (PointDataVector   &data     , cv::Scalar color, SensorTf & tf_r);

    template <class SegData>
    void plot_segm  (boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &data, cv::Scalar color, SensorTf & tf_r);
    template <class SegData>
    void plot_kalman(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > _input, cv::Scalar col_cov_x, cv::Scalar col_cov_v);

    void plot_t_bug(double d_followed, ObjectDataPtr o_followed, int dir_followed, polar target_p, polar target_p_ext, polar potential);

    //Constructors & Destructors
    PlotWorld(std::string wndView, RecfgParam &_param, SensorTf& _tf_sns, KalmanSLDM &_k);
    ~PlotWorld(){}

    SensorTf&   tf_sns;
    SensorTf    tf_r;
private:
    int         image_size;
    bool&       plot_world;
    bool&       plot_grid;
    double&     view_len;
    xy          view_center;
    KalmanSLDM& k;
    ObjColor    o_col[5];
    double      writing_to_bag;

    static void mouseCallBackWorld(int evt, int c, int r, int flags, void *param );
};

extern template void PlotWorld::plot_segm<SegmentData   >(SegmentDataPtrVectorPtr    &data, cv::Scalar color, SensorTf & tf_r);
extern template void PlotWorld::plot_segm<SegmentDataExt>(SegmentDataExtPtrVectorPtr &data, cv::Scalar color, SensorTf & tf_r);


#endif // PLOT_WORLD_H
