#include "utils/iterators_ss_tf.h"
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
IteratorIndexSet_ss_tf::IteratorIndexSet_ss_tf(const SegmentDataPtrVectorPtr &_input,IISmode mode){
    if(!_input){
        status_ = IIS_INVALID;
        return;
    }
    input_=_input;
    imax_=-1;
    for(SegmentDataPtrVectorIter seg_sum=input_->begin();seg_sum!=input_->end();seg_sum++){
        imax_+=(*seg_sum)->p_tf.size();
    }
    if(imax_ != -1){
        if(mode == FWD){
            seg_=input_->begin();
            p_=(*seg_)->p_tf.begin();
            i_=0;
        }
        else if( mode == REV ){
            seg_=--input_->end();
            p_=--(*seg_)->p_tf.end();
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
IISstatus IteratorIndexSet_ss_tf::update_status(){
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
            if( (*seg_)->p_tf.size() == 0 ){
                return status_=IIS_INVALID;
            }
            else{
                if(p_ == (*seg_)->p_tf.end()){
                    return status_=IIS_P_END;
                }
                else if( p_ == (*seg_)->p_tf.begin() ){
                    return status_=IIS_P_BEGIN;
                }
                else if( p_ == --(*seg_)->p_tf.end()){
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
bool IteratorIndexSet_ss_tf::advance(IISmode mode, IISmode dir){
    if( dir == INC ){
        if(( status_ == IIS_INVALID )||(status_ == IIS_SEG_END)){
            return false;
        }
        else if(status_ == IIS_SEG_REND){
            seg_=input_->begin();
            p_=--(*seg_)->p_tf.begin();
        }
        i_++;
        if( ++p_ == (*seg_)->p_tf.end() ){
            if( mode == ALL_SEGM ){
                if( ++seg_ == input_->end() ){
                    update_status();
                    return false;
                }
                else{
                    p_ = (*seg_)->p_tf.begin();
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
            p_=(*seg_)->p_tf.end();
        }
        i_--;
        if( --p_ == --(*seg_)->p_tf.begin() ){
            if( mode == ALL_SEGM ){
                if( --seg_ == --input_->begin() ){
                    update_status();
                    return false;
                }
                else{
                    p_ = --(*seg_)->p_tf.end();
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

void IteratorIndexSet_ss_tf::push_bk(SegmentDataPtrVectorPtr &output, PointData val,bool &split_segment){
    if(( seg_ != seg_old_ )||(split_segment)){
        output->push_back(SegmentDataPtr(new SegmentData(seg_,output->size())));
    }
    output->back()->p_tf.push_back(val);
    split_segment=false;
    seg_old_ = seg_;
    if( output == input_ ){
        imax_++;
        update_status();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void IteratorIndexSet_ss_tf::push_bk(SegmentDataPtrVectorPtr &output, PointData val){
    if(( seg_ != seg_old_ )){
        output->push_back(SegmentDataPtr(new SegmentData(seg_)));
    }
    output->back()->p_tf.push_back(val);
    seg_old_ = seg_;
    if( output == input_ ){
        imax_++;
        update_status();
    }
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

void IteratorIndexSet_ss_tf::pop_bk(SegmentDataPtrVectorPtr &output){
    if(output->size() >0 ){
        if(output->back()->p_tf.size() >0 ){
            output->back()->p_tf.pop_back();
        }
        if(output->back()->p_tf.size() == 0 ){
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

bool IteratorIndexSet_ss_tf::erase(){
    p_=(*seg_)->p_tf.erase(p_);
    if((*seg_)->p_tf.size() == 0){
        seg_=input_->erase(seg_);
        p_=(*seg_)->p_tf.begin();
    }
    imax_--;
    update_status();
}

///------------------------------------------------------------------------------------------------------------------------------------------------///

IteratorIndexSet2_ss_tf::IteratorIndexSet2_ss_tf(const SegmentDataPtrVectorPtr &_input,IISmode mode){
    maj_= IteratorIndexSet_ss_tf(_input,mode);
    min_= IteratorIndexSet_ss_tf(_input,mode);
    update_status();
}

IIS2status IteratorIndexSet2_ss_tf::update_status(){
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

bool IteratorIndexSet2_ss_tf::advance(IISmode mode, IISmode dir){
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

bool IteratorIndexSet2_ss_tf::advance_divergent(IteratorIndexSet_ss_tf &iis_out, double* ang_bounds){
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

bool IteratorIndexSet2_ss_tf::advance_in_ang_bounds(double* ang_bounds){
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

