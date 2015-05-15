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


#include "utils/iterators.h"
#include "utils/base_classes.h"
#include "utils/math.h"

using namespace std;


/**
 * Constructor
 *
 * @param _input input full list
 * @param mode FWD (iterators are initialized on first segment, first point,i = 0) or
 *             REV  (iterators are initialized on last segment , last point ,i = total number of points - 1)
 **/
template<class SegData>
IteratorIndexSet<SegData>::IteratorIndexSet(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, IISmode mode){
    if(!_input){
        status_ = IIS_INVALID;
        return;
    }
    input_=_input;
    imax_=-1;
    for(typename std::vector<boost::shared_ptr<SegData> >::iterator seg_sum=input_->begin();seg_sum!=input_->end();seg_sum++){
        imax_+=(*seg_sum)->p.size();
    }
    if(imax_ != -1){
        if(mode == FWD){
            seg_=input_->begin();
            p_=(*seg_)->p.begin();
            i_=0;
        }
        else if( mode == REV ){
            seg_=--input_->end();
            p_=--(*seg_)->p.end();
            i_=imax_;
        }
    }
    update_status();
}

/**
 * updates status of iterator :
 *
 * @return IIS_INVALID   : - list has 0 segments
 *                         - segment iterator has 0 points
 *         IIS_SEG_END   : - segment iterator is in end position
 *         IIS_SEG_REND  : - segment iterator is in rend position
 *         IIS_P_END     : - point iterator   is in end/rend position
 *         IIS_P_BEGIN   : - point iterator   is in first position
 *         IIS_P_RBEGIN  : - point iterator   is in last position
 *         IIS_VALID     : - else
 **/
template<class SegData>
IISstatus IteratorIndexSet<SegData>::update_status(){
    if( imax_ == -1 ){
        return status_=IIS_INVALID;
    }
    else{
        if( i_ == imax_ + 1 ){
            return status_=IIS_SEG_END;
        }
        else if( i_ == -1 ){
            return status_=IIS_SEG_REND;
        }
        else{
            if( (*seg_)->p.size() == 0 ){
                return status_=IIS_INVALID;
            }
            else{
                if(p_ == (*seg_)->p.end()){
                    return status_=IIS_P_END;
                }
                else if( p_ == (*seg_)->p.begin() ){
                    return status_=IIS_P_BEGIN;
                }
                else if( p_ == --(*seg_)->p.end()){
                    return status_=IIS_P_RBEGIN;
                }
            }
        }
    }
    return status_=IIS_VALID;
}

/**
 * advances iterators positions:
 *
 * - updates status of iterator after advancement
 * @param mode ALL_SEGM - iterates on entire segment list
 *             ONE_SEGM - iterates on points from given segment iterator
 * @param dir  INC - increments
 *             DEC - decrements
 * @return false - iterator is invalid
 *               - advanced segment iterator points to end/rend of list
 *         true  - else
 **/
template<class SegData>
bool IteratorIndexSet<SegData>::advance(IISmode mode, IISmode dir){
    if( dir == INC ){
        if(( status_ == IIS_INVALID )||(status_ == IIS_SEG_END)){
            return false;
        }
        else if(status_ == IIS_SEG_REND){
            seg_=input_->begin();
            p_=--(*seg_)->p.begin();
        }
        i_++;
        if( ++p_ == (*seg_)->p.end() ){
            if( mode == ALL_SEGM ){
                if( ++seg_ == input_->end() ){
                    update_status();
                    return false;
                }
                else{
                    p_ = (*seg_)->p.begin();
                }
            }
            else{
                update_status();
                return false;
            }
        }
    }
    else if( dir == DEC ){
        if(( status_ == IIS_INVALID )||(status_ == IIS_SEG_REND)){
            return false;
        }
        else if(status_ == IIS_SEG_END){
            seg_=--input_->end();
            p_=(*seg_)->p.end();
        }
        i_--;
        if( --p_ == --(*seg_)->p.begin() ){
            if( mode == ALL_SEGM ){
                if( --seg_ == --input_->begin() ){
                    update_status();
                    return false;
                }
                else{
                    p_ = --(*seg_)->p.end();
                }
            }
            else{
                update_status();
                return false;
            }
        }
    }
    update_status();
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
void IteratorIndexSet<SegData>::push_bk(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output, PointData val, bool &split_segment){
    if(( seg_ != seg_old_ )||(split_segment)){
        output->push_back(boost::shared_ptr<SegData>(new SegData(seg_,(int)output->size())));
    }
    output->back()->p.push_back(val);
    split_segment=false;
    seg_old_ = seg_;
    if( output == input_ ){
        imax_++;
        update_status();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
void IteratorIndexSet<SegData>::push_bk(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output, PointData val){
    if(( seg_ != seg_old_ )){
        output->push_back(boost::shared_ptr<SegData>(new SegData(seg_)));
    }
    output->back()->p.push_back(val);
    seg_old_ = seg_;
    if( output == input_ ){
        imax_++;
        update_status();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
void IteratorIndexSet<SegData>::pop_bk(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &output){
    if(output->size() >0 ){
        if(output->back()->p.size() >0 ){
            output->back()->p.pop_back();
        }
        if(output->back()->p.size() == 0 ){
            output->pop_back();
            seg_old_=--seg_old_;
        }
    }
    if( output == input_ ){
        imax_--;
        update_status();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
bool IteratorIndexSet<SegData>::erase(){
    p_=(*seg_)->p.erase(p_);
    if((*seg_)->p.size() == 0){
        seg_=input_->erase(seg_);
        p_=(*seg_)->p.begin();
    }
    imax_--;
    update_status();
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
IteratorIndexSet2<SegData>::IteratorIndexSet2(boost::shared_ptr<std::vector<boost::shared_ptr<SegData> > > &_input, IISmode mode){
    maj_= IteratorIndexSet<SegData>(_input,mode);
    min_= IteratorIndexSet<SegData>(_input,mode);
    update_status();
}
template<class SegData>
IIS2status IteratorIndexSet2<SegData>::update_status(){
    if(((min_.status() == IIS_INVALID )&&(maj_.status() == IIS_INVALID))||
       ((min_.status() == IIS_SEG_END )&&(maj_.status() == IIS_SEG_END))){
        return status_=IIS2_INVALID;
    }
    else if(min_.status() == IIS_SEG_REND){
        return status_=IIS2_MIN_SEG_REND;
    }
    else if(maj_.status() == IIS_SEG_END){
        return status_=IIS2_MAJ_SEG_END;
    }
    else if( maj_.seg() == min_.seg() ){
        return status_=IIS2_ONE_SEG;
    }
    else if( maj_.seg() != min_.seg() ){
        return status_=IIS2_TWO_SEG;
    }
    else{
        return status_=IIS2_VALID;
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
bool IteratorIndexSet2<SegData>::advance(IISmode mode, IISmode dir){
    if((( dir == INC )&&( maj_.status() == IIS_SEG_END )&&( min_.status() != IIS_SEG_END ))||
       (( dir == DEC )&&( min_.status() == IIS_SEG_END )&&( maj_.status() != IIS_SEG_END ))){
        return false;
    }
    maj_.advance(mode,dir);
    min_.advance(mode,dir);
    update_status();
    if(( mode == ONE_SEGM )&&(( maj_.status() == IIS_P_END )||( min_.status() == IIS_P_END ))){
            return false;
    }
    if( status_ < IIS2_VALID  ){
        return false;
    }
    if(( min_.status() < IIS_VALID )&&(maj_.status() < IIS_VALID)){
        return false;
    }
    return true;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
bool IteratorIndexSet2<SegData>::advance_divergent(IteratorIndexSet<SegData> &iis_out, double* ang_bounds){
    static bool flip_inc   = true;
    flip_inc = !flip_inc;
    if(((min_.status() >= IIS_VALID)&&( flip_inc ))||((min_.status() >= IIS_VALID)&&(maj_.status() < IIS_VALID))){
        min_.advance(ALL_SEGM,DEC);

        if(min_.status() < IIS_VALID){
            return advance_divergent(iis_out,ang_bounds);
        }
        else if(min_.p()->angle < ang_bounds[0]){
            min_.status_ = IIS_HARD_INVALID;
            return advance_divergent(iis_out,ang_bounds);
        }
        else{
            iis_out = min_;update_status();return true;
        }
    }
    if(((maj_.status() >= IIS_VALID)&&( !flip_inc ))||((maj_.status() >= IIS_VALID)&&(min_.status() < IIS_VALID))){
        maj_.advance(ALL_SEGM,INC);

        if(maj_.status() < IIS_VALID){
            return advance_divergent(iis_out,ang_bounds);
        }
        else if(maj_.p()->angle > ang_bounds[1]){
            maj_.status_ = IIS_HARD_INVALID;
            return advance_divergent(iis_out,ang_bounds);
        }
        else{
            iis_out = maj_;update_status();return true;
        }
    }
    flip_inc = true;
    maj_.update_status();min_.update_status();
    update_status();
    if( status_ == IIS2_MAJ_SEG_END ){
        maj_.advance(ALL_SEGM,DEC);
    }
    if( status_ == IIS2_MIN_SEG_REND ){
        maj_.advance(ALL_SEGM,INC);
    }
    return false;
}

///------------------------------------------------------------------------------------------------------------------------------------------------///
template<class SegData>
bool IteratorIndexSet2<SegData>::advance_in_ang_bounds(double* ang_bounds){
    update_status();
    if((maj_.status() == IIS_SEG_END)&&(min_.status() == IIS_SEG_REND)){
        min_.advance(ALL_SEGM,INC);
        maj_ = min_;
        update_status();
    }
    else if(maj_.status() == IIS_SEG_END){
        maj_ = min_;update_status();
    }
    else if(min_.status() == IIS_SEG_REND){
        min_ = maj_;update_status();
    }
    if(!( ( ( min_.p()->angle > ang_bounds[1] ) || ( min_.p()->angle < ang_bounds[0] ) ) &&
          ( ( maj_.p()->angle > ang_bounds[1] ) || ( maj_.p()->angle < ang_bounds[0] ) ) )){//if NOT both are outside search area

        if     (( min_.p()->angle > ang_bounds[1] ) || ( min_.p()->angle < ang_bounds[0] )){//if maj is in search area
            min_ = maj_;update_status();return true;
        }
        else if(( maj_.p()->angle > ang_bounds[1] ) || ( maj_.p()->angle < ang_bounds[0] )){//if min is in search area
            maj_ = min_;update_status();return true;
        }
        else{//if both in search area, choose to merge in closest to the search angle centre
            double angl_avg=(ang_bounds[1] + ang_bounds[0]) / 2.0;
            if(( fabs( min_.p()->angle - angl_avg ) < fabs( maj_.p()->angle - angl_avg ) )){
                maj_ = min_;update_status();return true;
            }
            else{
                min_ = maj_;update_status();return true;
            }
        }
    }
    double d_angle_min = 100.0;bool merge_to_min; int pos;

    if       ( min_.p()->angle - ang_bounds[1] >= 0 )                                       { d_angle_min = min_.p()->angle - ang_bounds[1]; merge_to_min =  true; pos = -1;}
    else if  ( ang_bounds[0] - min_.p()->angle >  0 )                                       { d_angle_min = ang_bounds[0] - min_.p()->angle; merge_to_min =  true; pos =  1;}
    if     ( ( maj_.p()->angle - ang_bounds[1] >  0 ) && ( maj_.p()->angle < d_angle_min ) ){ d_angle_min = maj_.p()->angle - ang_bounds[0]; merge_to_min = false; pos = -1;}
    else if( ( ang_bounds[0] - maj_.p()->angle >= 0 ) && ( maj_.p()->angle < d_angle_min ) ){ d_angle_min = ang_bounds[0] - maj_.p()->angle; merge_to_min = false; pos =  1;}

    if( merge_to_min ){ maj_ = min_;}
    else              { min_ = maj_;}

    do{}while(advance(ALL_SEGM,(IISmode)max(pos,0)) && ( (pos * ( ang_bounds[std::max(-pos,0)] - min_.p()->angle ) > 0) ) );

    update_status();
    if(( min_.status() < IIS_VALID )&&(maj_.status() < IIS_VALID)){
        return false;
    }
    return true;
}

template class IteratorIndexSet <SegmentData>;
template class IteratorIndexSet <SegmentDataExt>;
template class IteratorIndexSet2<SegmentData>;
template class IteratorIndexSet2<SegmentDataExt>;
