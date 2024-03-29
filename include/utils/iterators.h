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

#ifndef ITERATORS_H
#define ITERATORS_H
#include "utils/base_classes.h"

class SegData;

template<class SegData>
class IteratorIndexSet2;


enum IISstatus{
    IIS_HARD_INVALID=0, //user-issued invalid
    IIS_INVALID=1,      //iterator in 0-elements-input or in 0-elements-segment
    IIS_SEG_END=2,      //iterator in segment end
    IIS_SEG_REND=3,     //iterator in regment rend
    IIS_P_END=4,        //iterator in p end or rend
    IIS_P_REND=5,       //iterator in p rend
    IIS_VALID=6,        //iterator in  valid position
    IIS_P_BEGIN=7,      //iterator in (valid) p begin position
    IIS_P_RBEGIN=8      //iterator in (valid) p rbegin position
};

enum IIS2status{
    IIS2_HARD_INVALID=0, //user-issued invalid
    IIS2_INVALID=1,      //both iterators are SEG_END,SEG_REND or INVALID   
    IIS2_VALID=2,        //iterator in  valid position
    IIS2_MAJ_SEG_END=3,  //maj iterator is in SEG_END
    IIS2_MIN_SEG_REND=4, //min iterator is in SEG_REND
    IIS2_ONE_SEG=5,      //both iterators in the same segment
    IIS2_TWO_SEG=6       //iterators in diferent segments
};

enum IISmode{
    INC=1,
    DEC=0,
    FWD=1,
    REV=0,
    ONE_SEGM=2,
    ALL_SEGM=3
};

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
class IteratorIndexSet{
public:

    friend class IteratorIndexSet2<SegData>;

    boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > >&   input() {return input_;}
    typename std::vector<boost::shared_ptr<SegData> >::iterator&  seg()   {return seg_;}
    PointDataVectorIter&   p()     {return p_;}
    int&                   i()     {return i_;}
    int&                   imax()  {return imax_;}
    IISstatus&             status(){return status_;}

    IISstatus update_status();
    bool advance(IISmode mode, IISmode dir);
    void push_bk(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output, PointData val, bool &split_segment);
    void push_bk(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output, PointData val);
    void pop_bk (boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output);
    bool erase  ();
    //void erase  (boost::shared_ptr<std::Vector<SegmentData> > &output,std::Vector<SegmentData>::iterator _seg);

    //Constructors & Destructors
    IteratorIndexSet(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, IISmode mode=FWD);
    IteratorIndexSet(){}
    ~IteratorIndexSet(){}
private:

    boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > >  input_;
    typename std::vector<boost::shared_ptr<SegData> >::iterator seg_;
    PointDataVectorIter   p_;
    int                 i_;
    int                 imax_;
    IISstatus           status_;
    typename std::vector<boost::shared_ptr<SegData> >::iterator seg_old_;

};

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
class IteratorIndexSet2{
public:

    IteratorIndexSet<SegData>& maj()  {return maj_;}
    IteratorIndexSet<SegData>& min()  {return min_;}
    IIS2status&       status(){return status_;}

    bool advance(IISmode mode, IISmode dir);
    bool advance_divergent(IteratorIndexSet<SegData> &iis_out, double* ang_bounds=NULL);
    bool advance_in_ang_bounds(double* ang_bounds);

    //Constructors & Destructors
    IteratorIndexSet2(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, IISmode mode=FWD);
    IteratorIndexSet2(){}
    ~IteratorIndexSet2(){}
private:

    IteratorIndexSet<SegData> maj_;
    IteratorIndexSet<SegData> min_;
    IIS2status       status_;
    IIS2status update_status();
};

#endif // ITERATORS_H
