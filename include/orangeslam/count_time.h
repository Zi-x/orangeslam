#pragma once
#ifndef ORANGESLAM_COUNT_TIME_H
#define ORANGESLAM_COUNT_TIME_H
#include "orangeslam/common_include.h"
namespace orangeslam{
struct count_time{

    int step_count_ = 0;
    double total_time_ = 0.0;
    double min_frame_rate_ = 999999;
    double max_frame_rate_ = 0.0;
    std::vector<double> frame_rates_;
    

    void RecordFrameRate(double frame_rate) {
        frame_rates_.push_back(frame_rate);
        
        if (frame_rate < min_frame_rate_) {
            min_frame_rate_ = frame_rate;
        }
        if (frame_rate > max_frame_rate_) {
            max_frame_rate_ = frame_rate;
        }
    }
    
    double GetAverageFrameRate() const {
        double sum = 0.0;
        for (double frame_rate : frame_rates_) {
            sum += frame_rate;
        }
        
        return frame_rates_.empty() ? 0.0 : sum / frame_rates_.size();
    }
    
};
}
#endif
