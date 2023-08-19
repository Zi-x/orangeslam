#pragma once
#ifndef ORANGESLAM_FRONTEND_H
#define ORANGESLAM_FRONTEND_H

#include <opencv2/features2d.hpp>

#include "orangeslam/common_include.h"
#include "orangeslam/frame.h"
#include "orangeslam/map.h"
#include "orangeslam/count_time.h"

namespace orangeslam {

class Backend;
class Viewer;
class Fdetect;
enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
 * 前端
 * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
 */
class Frontend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    /// 外部接口，添加一个帧并计算其定位结果
    bool AddFrame(Frame::Ptr frame);

    /// Set函数
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; if_set_viewer = true; }

    void setFdetect(std::shared_ptr<Fdetect> fdetect){ fdetect_ = fdetect; if_set_fdetect = true; }
    
    FrontendStatus GetStatus() const { return status_; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

    bool inBorder(const cv::Point2f &pt)
        {
        const int BORDER_SIZE = 0.2;
        int img_x = cvRound(pt.x);
        int img_y = cvRound(pt.y);
        float w = camera_left_->width;
        float h = camera_left_->height;
        return BORDER_SIZE <= img_x && img_x < w - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < h - BORDER_SIZE;
        }

    count_time countfor_tracklast;
    count_time countfor_estimate;

    count_time countfor_detect;
    count_time countfor_findR;
    count_time countfor_trian;

    count_time countfor_back_update;
    count_time countfor_view_update;



   private:
    /**
     * Track in normal mode
     * @return true if success
     */
    bool Track();

    /**
     * Reset when lost
     * @return true if success
     */
    bool Reset();

    /**
     * Track with last frame
     * @return num of tracked points
     */
    int TrackLastFrame();

    /**
     * estimate current frame's pose
     * @return num of inliers
     */
    int EstimateCurrentPose();

    /**
     * set current frame as a keyframe and insert it into backend
     * @return true if success
     */
    bool InsertKeyframe();

    /**
     * Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     */
    bool StereoInit();

    /**
     * Detect features in left image in current_frame_
     * keypoints will be saved in current_frame_
     * @return
     */
    int DetectFeatures();

    /**
     * Find the corresponding features in right image of current_frame_
     * @return num of features found
     */
    int FindFeaturesInRight();

    /**
     * Build the initial map with single image
     * @return true if succeed
     */
    bool BuildInitMap();

    /**
     * Triangulate the 2D points in current frame
     * @return num of triangulated points
     */
    int TriangulateNewPoints();

    /**
     * Set the features in keyframe as new observation of the map points
     */
    void SetObservationsForKeyFrame();

    // set those in default.yaml
    bool if_set_viewer = false;
    bool if_set_fdetect = false;
    bool if_set_inborder = false;

    int detect_max_features_; 
    int num_features_init_ = 50; 

    int xgrid = 3;
    int ygrid = 2;

    // data
    FrontendStatus status_ = FrontendStatus::INITING;

    Frame::Ptr current_frame_ = nullptr;  // 当前帧
    Frame::Ptr last_frame_ = nullptr;     // 上一帧
    Camera::Ptr camera_left_ = nullptr;   // 左侧相机
    Camera::Ptr camera_right_ = nullptr;  // 右侧相机

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;
    std::shared_ptr<Fdetect> fdetect_ = nullptr;

    SE3 relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧pose初值

    int tracking_inliers_ = 0;  // inliers, used for testing new keyframes

    //disparity
    float average_one_track_disparity_x = 0;
    float average_one_track_disparity_y = 0;
    float disparity_accumulate_x = 0;
    float disparity_accumulate_y = 0;
    float disparity_accumulate_xy = 0;
    float disparity_needed_for_keyframe = 30; //视差阈值
    float inlier_features_needed_for_not_keyframe_ = 160;

    // params
    int num_features_tracking_ = 50; //50
    int num_features_tracking_bad_ = 20; //20
    int inlier_features_needed_for_keyframe_ = 85;//85
    
    // utilities
    // cv::Ptr<cv::GFTTDetector> gftt_;  // feature detector in opencv
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
};

}  // namespace orangeslam

#endif  // ORANGESLAM_FRONTEND_H
