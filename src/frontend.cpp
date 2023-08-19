#include <opencv2/opencv.hpp>
#include "orangeslam/algorithm.h"
#include "orangeslam/backend.h"
#include "orangeslam/config.h"
#include "orangeslam/feature.h"
#include "orangeslam/frontend.h"
#include "orangeslam/g2o_types.h"
#include "orangeslam/map.h"
#include "orangeslam/viewer.h"
#include "orangeslam/fdetect.h"
namespace orangeslam {

Frontend::Frontend() {
    // gftt_ =
    //     cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    int threshold = 25; // FAST 角点阈值
    // 是否进行非极大值抑制
    fast_detector = cv::FastFeatureDetector::create(threshold, true);
    num_features_init_ = Config::Get<int>("num_features_init");
    detect_max_features_ = Config::Get<int>("detect_max_features");
    xgrid = Config::Get<int>("grid_size_x");
    ygrid = Config::Get<int>("grid_size_y");
    if(Config::Get<int>("if_set_inborder")){
        if_set_inborder = true;
    }

    
}

bool Frontend::AddFrame(orangeslam::Frame::Ptr frame) {
    current_frame_ = frame;
    
    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }
    last_frame_ = current_frame_;
    return true;
    }
bool Frontend::Track() {
    LOG(INFO) << "notify lock in Track ";
    if(if_set_fdetect) fdetect_->DetectCurrentFrame(current_frame_);
    
    if (last_frame_) {
        // 假设过程A-B-C,用A-B近似B-C，乘以上一帧的位姿，设置预估的current_frame_的pose，将预估下一帧的初值传入EstimateCurrentPose
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    
    TrackLastFrame();
    disparity_accumulate_x += average_one_track_disparity_x;
    disparity_accumulate_y += average_one_track_disparity_y;
    disparity_accumulate_xy = sqrt(disparity_accumulate_x * disparity_accumulate_x + disparity_accumulate_y * disparity_accumulate_y);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        countfor_tracklast.step_count_++;
        countfor_tracklast.total_time_ += time_used;

    if (countfor_tracklast.total_time_ >= 49.0) {
        double average_time_tracklast = countfor_tracklast.total_time_ / countfor_tracklast.step_count_;
        countfor_tracklast.RecordFrameRate(average_time_tracklast);
        countfor_tracklast.step_count_ = 0;
        countfor_tracklast.total_time_ = 0.0;
    }

    // 3D-2D其中3D点是三角化得到的，都是基于世界坐标系的points
    // 所以估计得到的相机pose都是基于世界坐标系(第一帧)的
    tracking_inliers_ = EstimateCurrentPose();
    auto t3 = std::chrono::high_resolution_clock::now();
    time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    countfor_estimate.step_count_++;
    countfor_estimate.total_time_ += time_used;
    
    if (countfor_estimate.total_time_ >= 49.0) {
        double average_time_estimate = countfor_estimate.total_time_ / countfor_estimate.step_count_ ;
        countfor_estimate.RecordFrameRate(average_time_estimate);
        countfor_estimate.step_count_ = 0;
        countfor_estimate.total_time_ = 0.0;
    }

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking badadsadadsasd
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }   


    if(if_set_fdetect)
    if(tracking_inliers_ <= inlier_features_needed_for_keyframe_ * 1.5){
        auto s1_time = std::chrono::high_resolution_clock::now();
        if(fdetect_->fdetect_completed_bool_ == false)
        {
            std::mutex mutex;
            std::unique_lock<std::mutex> lock(mutex);
            LOG(INFO) << "wait lock in frontend ";
            if (fdetect_->fdetect_completed_bool_ == false) fdetect_->fdetect_completed_condition_.wait(lock);
        }
        auto s2_time = std::chrono::high_resolution_clock::now();
        auto used_time = std::chrono::duration_cast<std::chrono::milliseconds>(s2_time - s1_time ).count();
        LOG(INFO) << "fdetect_result wait time used: "<< used_time << " ms";

    }
    

    bool insert = InsertKeyframe();
    // 计算相对位姿A-B
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
    // if(if_set_viewer)
    if (viewer_ && !insert) viewer_->AddCurrentFrame(current_frame_);
    
    return true;
}

bool Frontend::InsertKeyframe() {
    if ((tracking_inliers_ >= inlier_features_needed_for_keyframe_ && disparity_accumulate_xy <= disparity_needed_for_keyframe)||
        tracking_inliers_ >= inlier_features_needed_for_not_keyframe_) { 
        // Don't insert keyframe
        return false;
    }
    LOG(INFO) << "disparity_accumulate_xy: " << disparity_accumulate_xy;
    LOG(INFO) << "tracking_inliers_: " << tracking_inliers_;
    disparity_accumulate_x = 0;
    disparity_accumulate_y = 0;

    auto t3_ik = std::chrono::high_resolution_clock::now();
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    
    map_->InsertKeyFrame(current_frame_);
    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();
    
    DetectFeatures();  // detect new features
    auto t4 = std::chrono::high_resolution_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3_ik).count();
    countfor_detect.step_count_++;
    countfor_detect.total_time_ += time_used;
    if (countfor_detect.total_time_ >= 49.0) {
        double average_time_detect = countfor_detect.total_time_ / countfor_detect.step_count_;
        countfor_detect.RecordFrameRate(average_time_detect);
        countfor_detect.step_count_ = 0;
        countfor_detect.total_time_ = 0.0;
    } 
    // track in right image
    FindFeaturesInRight();
    auto t5 = std::chrono::high_resolution_clock::now();
    time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

    countfor_findR.step_count_++;
    countfor_findR.total_time_ += time_used;
    if (countfor_findR.total_time_ >= 49.0) {
        double average_time_findR = countfor_findR.total_time_ / countfor_findR.step_count_;
        countfor_findR.RecordFrameRate(average_time_findR);
        countfor_findR.step_count_ = 0;
        countfor_findR.total_time_ = 0.0;
    } 

    // triangulate map points
    TriangulateNewPoints();
    auto t6 = std::chrono::high_resolution_clock::now();
    time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

    countfor_trian.step_count_++;
    countfor_trian.total_time_ += time_used;
    if (countfor_trian.total_time_ >= 49.0) {
        double average_time_trian = countfor_trian.total_time_ / countfor_trian.step_count_;
        countfor_trian.RecordFrameRate(average_time_trian);
        countfor_trian.step_count_ = 0;
        countfor_trian.total_time_ = 0.0;
    } 
    
    // update backend because we have a new keyframe
    /// InsertKeyFrame两者有前后关系，不用上线程锁
    /// 但若上一帧一直到现在还没优化完，这里InsertKeyFrame是否会影响
    /// 后端优化线程的安全？是否应该上锁？
    backend_->UpdateMap();

    auto t7 = std::chrono::high_resolution_clock::now();
    time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();

    countfor_back_update.step_count_++;
    countfor_back_update.total_time_ += time_used;
    if (countfor_back_update.total_time_ >= 49.0) {
        double average_time_back = countfor_back_update.total_time_ / countfor_back_update.step_count_;
        countfor_back_update.RecordFrameRate(average_time_back);
        countfor_back_update.step_count_ = 0;
        countfor_back_update.total_time_ = 0.0;
    } 
    // if(if_set_viewer)
    if (viewer_) viewer_->UpdateMap();

    auto t8 = std::chrono::high_resolution_clock::now();
    time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count();

    countfor_view_update.step_count_++;
    countfor_view_update.total_time_ += time_used;
    if (countfor_view_update.total_time_ >= 49.0) {
        double average_time_view = countfor_view_update.total_time_ / countfor_view_update.step_count_;
        countfor_view_update.RecordFrameRate(average_time_view);
        countfor_view_update.step_count_ = 0;
        countfor_view_update.total_time_ = 0.0;
    } 
    
    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0 ) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    // typedef g2o::LinearSolverEigen
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 2.991;
    int cnt_outlier = 0;
    
    for (int iteration = 0; iteration < 4; ++iteration) {
        
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
 
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    if(if_set_inborder)
    for (int i = 0; i < int(kps_current.size()); i++){
        if (status[i] && !inBorder(kps_current[i])){
            status[i] = 0;
            LOG(INFO) << "不在边界内";
        }
        
    }

    int num_good_pts = 0;
    int disparity_x_sum = 0;
    int disparity_y_sum = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            float disparity_x = kps_current[i].x - kps_last[i].x;
            float disparity_y = kps_current[i].y - kps_last[i].y;

            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));

            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            disparity_x_sum += disparity_x;
            disparity_y_sum += disparity_y;
            num_good_pts++;
        }
    }
    average_one_track_disparity_x = disparity_x_sum / static_cast<float>(num_good_pts);
    average_one_track_disparity_y = disparity_y_sum / static_cast<float>(num_good_pts);
    // LOG(INFO) << "average_one_track_disparity_x: " << average_one_track_disparity_x;
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    // fdetect_初始化需要一点时间，等待其进去wait再DetectCurrentFrame
    std::this_thread::sleep_for(std::chrono::milliseconds(52));
    LOG(INFO) << "notify lock in StereoInit ";
    if(if_set_fdetect) fdetect_->DetectCurrentFrame(current_frame_); 

    if(if_set_fdetect)
    if(fdetect_->fdetect_completed_bool_ == false)
    {
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        // if fdetect_->fdetect_completed_bool_ == false
        LOG(INFO) << "wait lock in StereoInit ";
        fdetect_->fdetect_completed_condition_.wait(lock);
        //fdetect_->fdetect_condition_completed_.wait(lock, [this](){ return fdetect_->operation_completed; });
    
    }


    DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        // if(if_set_viewer)
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    int pre_feat_num = 0;
    for (auto &feat : current_frame_->features_left_) {
        pre_feat_num++;
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }
    if(if_set_fdetect)
    for (auto &fresult : fdetect_->fdetect_result) {
            cv::rectangle(mask, fresult.first,
                        fresult.second, 0, CV_FILLED);
    }

    

    // std::vector<cv::Point2f> points;
    // cv::goodFeaturesToTrack(current_frame_->left_img_, points, detect_max_features_ ,0.01, 20, mask);

    // code referenced with https://zhuanlan.zhihu.com/p/591111705
    std::vector<cv::KeyPoint> keypoints, gridKeyPoints;
    // 网格的尺寸
    int xsize(current_frame_->left_img_.cols/xgrid), ysize(current_frame_->left_img_.rows/ygrid);
    // 每个网格需要提取多少高质量关键点
    int gridKeypointNum = detect_max_features_ / (xgrid * ygrid);
    // 网格图像,mask
    cv::Mat gridMat;
    cv::Mat gridMatMask;
    for (int y = 0; y < ygrid; y++)
        for (int x = 0; x < xgrid; x++)
        {
            // 某网格特征点
            gridKeyPoints.clear();
            gridMat = current_frame_->left_img_(cv::Rect(x*xsize, y*ysize, xsize, ysize));
            gridMatMask = mask(cv::Rect(x*xsize, y*ysize, xsize, ysize));
            fast_detector->detect(gridMat, gridKeyPoints, gridMatMask);

            // 截取较强的特征
            // 指向 gridKeyPoints 容器的末尾位置
            auto itEnd(gridKeyPoints.end());


            if(int(gridKeyPoints.size()) > gridKeypointNum)
            {
                // 排降序

                std::nth_element(gridKeyPoints.begin(),
                                 gridKeyPoints.begin() + gridKeypointNum,
                                 gridKeyPoints.end(),
                                 [](cv::KeyPoint& a, cv::KeyPoint& b){
                    return a.response > b.response;
                });
                // 截取较强的特征
                itEnd = gridKeyPoints.begin() + gridKeypointNum;
            }




            // 网格坐标-->整个图像坐标
            for (auto it = gridKeyPoints.begin(); it !=  itEnd; it++)
            {
                it->pt += cv::Point2f(x*xsize, y*ysize); // 注意是cv::Point2f
                keypoints.push_back(*it);
            }
        }
    // old method
    // std::vector<cv::KeyPoint> keypoints;
    // fast_detector->detect(current_frame_->left_img_, keypoints, mask);
    // if (keypoints.size() > detect_max_features_) {
    //     std::sort(keypoints.begin(), keypoints.end(),
    //             [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
    //                 return a.response > b.response;
    //             });
    //     keypoints.resize(detect_max_features_);
    // }
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
    
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        auto mp = kp->map_point_.lock();
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    if(if_set_inborder)
    for (int i = 0; i < int(kps_right.size()); i++){
        if (status[i] && !inBorder(kps_right[i])){
            status[i] = 0;
            LOG(INFO) << "不在right边界内";
        }
        
    }

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace orangeslam