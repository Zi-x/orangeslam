#pragma once
#ifndef ORANGESLAM_VISUAL_ODOMETRY_H
#define ORANGESLAM_VISUAL_ODOMETRY_H

#include "orangeslam/backend.h"
#include "orangeslam/common_include.h"
#include "orangeslam/dataset.h"
#include "orangeslam/frontend.h"
#include "orangeslam/viewer.h"
#include "orangeslam/fdetect.h"
#include "orangeslam/count_time.h"
namespace orangeslam {

/**
 * VO 对外接口
 */
class VisualOdometry {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

   private:

    // init
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // fdetect
    Fdetect::Ptr fdetect_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
    count_time countfor_vo;
};
}  // namespace orangeslam

#endif  // ORANGESLAM_VISUAL_ODOMETRY_H
