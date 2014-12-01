#ifndef ITERATORS_SS_TF_H
#define ITERATORS_SS_TF_H
#include "utils/base_classes.h"
#include "utils/iterators.h"

class IteratorIndexSet_ss_tf{
public:
    friend class IteratorIndexSet2_ss_tf;

    SegmentDataPtrVectorPtr&  input() {return input_;}
    SegmentDataPtrVectorIter& seg()   {return seg_;}
    PointDataVectorIter&   p()     {return p_;}
    int&                   i()     {return i_;}
    int&                   imax()  {return imax_;}
    IISstatus&             status(){return status_;}

    IISstatus update_status();
    bool advance(IISmode mode, IISmode dir);
    void push_bk(SegmentDataPtrVectorPtr &output, PointData val, bool &split_segment);
    void push_bk(SegmentDataPtrVectorPtr &output, PointData val);
    void pop_bk (SegmentDataPtrVectorPtr &output);
    bool erase  ();
    //void erase  (boost::shared_ptr<std::Vector<SegmentData> > &output,std::Vector<SegmentData>::iterator _seg);

    IteratorIndexSet_ss_tf(const SegmentDataPtrVectorPtr &_input, IISmode mode=FWD);
    IteratorIndexSet_ss_tf(){}
private:
    SegmentDataPtrVectorPtr  input_;
    SegmentDataPtrVectorIter seg_;
    PointDataVectorIter   p_;
    int                 i_;
    int                 imax_;
    IISstatus           status_;
    SegmentDataPtrVectorIter seg_old_;

};

///------------------------------------------------------------------------------------------------------------------------------------------------///

class IteratorIndexSet2_ss_tf{
public:
    IteratorIndexSet_ss_tf& maj()  {return maj_;}
    IteratorIndexSet_ss_tf& min()  {return min_;}
    IIS2status&       status(){return status_;}

    bool advance(IISmode mode, IISmode dir);
    bool advance_divergent(IteratorIndexSet_ss_tf &iis_out, double* ang_bounds=NULL);
    bool advance_in_ang_bounds(double* ang_bounds);

    IteratorIndexSet2_ss_tf(const SegmentDataPtrVectorPtr &_input, IISmode mode=FWD);
    IteratorIndexSet2_ss_tf(){}
private:
    IteratorIndexSet_ss_tf maj_;
    IteratorIndexSet_ss_tf min_;
    IIS2status       status_;
    IIS2status update_status();
};

#endif // ITERATORS_SS_H
