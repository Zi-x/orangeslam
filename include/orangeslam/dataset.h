#pragma once
#ifndef ORANGESLAM_DATASET_H
#define ORANGESLAM_DATASET_H
#include "orangeslam/config.h"
#include "orangeslam/camera.h"
#include "orangeslam/common_include.h"
#include "orangeslam/frame.h"
#include <opencv2/opencv.hpp>
namespace orangeslam {

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 */
class Dataset {
   public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    typedef std::shared_ptr<Dataset> Ptr;

    Dataset(const std::string& dataset_path);

    bool Init();

    /// create and return the next frame containing the stereo images
    Frame::Ptr DatasetNextFrame();

    Frame::Ptr RealworldNextFrame();

    bool initRealworldCamera();

    /// get camera by id
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

   private:
    cv::VideoCapture cap;

    cv::Mat map1_left, map2_left;
    
    cv::Mat map1_right, map2_right;

    cv::Mat distortionCoefficients_left;
    
    cv::Mat distortionCoefficients_right;

    cv::Mat openloris_cameraMatrix_right;

    cv::Mat openloris_cameraMatrix_left;

    cv::Mat realworld_cameraMatrix_right;

    cv::Mat realworld_cameraMatrix_left;
    
    std::string dataset_path_;

    SE3 T_openloris;

    SE3 T_realworld;

    std::vector<double> time_stamp_values;

    std::vector<double> time_stamp_values_left;

    std::vector<double> time_stamp_values_right;

    std::vector<std::string> openloris_filenames_right;
    
    std::vector<std::string> openloris_filenames_left;

    int current_image_index_ = 0;
    int current_timestamp_index_ = 0;

    float resize_scale;


    std::vector<Camera::Ptr> cameras_;
};
}  // namespace orangeslam

#endif