#pragma once
#ifndef ORANGESLAM_MAPPOINT_H
#define ORANGESLAM_MAPPOINT_H

#include "orangeslam/common_include.h"

namespace orangeslam {

struct Frame;

struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
struct MapPoint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0;  // ID
    bool is_outlier_ = false;
    Vec3 pos_ = Vec3::Zero();  // Position in world
    int observed_times_ = 0;  // being observed by feature matching algo.
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}

    MapPoint(long id, Vec3 position);

    Vec3 Pos() {
        std::unique_lock<std::mutex> lck(data_mutex_mappoint_);
        return pos_;
    }

    void SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_mappoint_);
        pos_ = pos;
    };

    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_mappoint_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_mappoint_);
        return observations_;
    }
    // factory function
    static MapPoint::Ptr CreateNewMappoint();

    private:
     std::mutex data_mutex_mappoint_;
};
}  // namespace orangeslam

#endif  // ORANGESLAM_MAPPOINT_H
