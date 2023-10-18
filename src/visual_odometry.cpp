//
// Created by gaoxiang on 19-5-4.
//
#include <iomanip>
#include "orangeslam/visual_odometry.h"
#include <chrono>
#include "orangeslam/config.h"


namespace orangeslam {
VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }
    if(Config::Get<int>("use_kitti")){
        dataset_dir_str = Config::Get<std::string>("dataset_dir_kitti");
        dataset_ = Dataset::Ptr(new Dataset(dataset_dir_str));
    }else if(Config::Get<int>("use_openloris")){
        dataset_dir_str = Config::Get<std::string>("dataset_dir_openloris");
        dataset_ = Dataset::Ptr(new Dataset(dataset_dir_str));
    }else if(Config::Get<int>("use_realworld")){
        dataset_ = Dataset::Ptr(new Dataset("realworld"));
        use_realworld_flag = true;
    }
    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    frontend_ = Frontend::Ptr(new Frontend);
    if(Config::Get<int>("if_fdetect_open")){
        fdetect_ = Fdetect::Ptr(new Fdetect);
    }
    if(Config::Get<int>("if_viewer_open")){
        viewer_ = Viewer::Ptr(new Viewer);
    }
    
    // frontend_ backend viewer setmap后内部的成员的map都指向map_,因此是共享的?
    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    
    if(Config::Get<int>("if_fdetect_open")){
        frontend_->setFdetect(fdetect_);
        
    }
    if(Config::Get<int>("if_viewer_open")){
        frontend_->SetViewer(viewer_);
    }

    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    if(Config::Get<int>("if_viewer_open")){
        viewer_->SetMap(map_);
    }

    if(Config::Get<int>("use_realworld")){
        dataset_->initRealworldCamera();
    }
    
    return true; 
}

void VisualOdometry::Run() {
    
    bool flag_close_viewer = false;
    while (1) {
        // LOG(INFO) << "VO is running";
        if (Step() == false || viewer_->GetCloseFlag()) {
        backend_->Stop();
        // fdetect_->Close();
        if(Config::Get<int>("if_saves_poses")){
            std::time_t now = std::time(nullptr);
            struct std::tm timeinfo;
            localtime_r(&now, &timeinfo);
            // 格式化时间为字符串，作为文件名的一部分
            char time_buffer[80];
            std::strftime(time_buffer, sizeof(time_buffer), "%m%d_%H%M", &timeinfo);
            // 构建序列名
            std::string last_two_chars;
            if (dataset_dir_str.size() >= 2) {
                last_two_chars = dataset_dir_str.substr(dataset_dir_str.size() - 2);
            } else {
                if(Config::Get<int>("use_realworld")) last_two_chars = "rw";
                LOG(INFO) << "The input string is too short.";
            }
            // 构建文件名
            std::string file_name;
            if(Config::Get<int>("set_saves_poses_kitti"))  file_name = "kitti_poses_" + last_two_chars +"_" + std::string(time_buffer) + ".txt";
            else if (Config::Get<int>("set_saves_poses_tum"))
            {
                file_name = "tum_poses_" + last_two_chars + "_" + std::string(time_buffer) + ".txt";
            }
            else{
                file_name = "tum_poses_" + last_two_chars +"_"  + std::string(time_buffer) + ".txt";
            }
            
            std::ofstream outputFile(file_name);
            std::unordered_map<unsigned long, Frame::Ptr> all_keyframes_;
            all_keyframes_ = map_->GetAllKeyFrames();
            for (size_t i = 0; i < all_keyframes_.size(); i++) {
            auto it = all_keyframes_.find(i);
            if (it != all_keyframes_.end()) {
                auto element = it->second;
                SE3 Twc;
                cv::Mat transMat;
                if(Config::Get<int>("use_openloris")){
                    if(Config::Get<std::string>("use_openloris_scene") == "corridor1"){
                        transMat = Config::Get<cv::Mat>("corridor1.trans_matrix.matrix_fisheye1_to_base_link");
                    }else if(Config::Get<std::string>("use_openloris_scene") == "market1"){
                        transMat = Config::Get<cv::Mat>("market1.trans_matrix.matrix_fisheye1_to_base_link");
                    }
                    
                    Eigen::Matrix4d eigentransMat;
                    for(int i=0;i<4;i++)
                        for(int j=0;j<4;j++)
                            eigentransMat(i,j) = transMat.at<double>(i,j);
                    SE3 T_eigentransMat = SE3(eigentransMat);
                    Twc = T_eigentransMat * element->Pose().inverse() ; //
                }else{
                    
                    Twc = element->Pose().inverse(); //;
                }
                
                Eigen::Vector3d translation = Twc.translation();
                Eigen::Matrix3d rotation_matrix = Twc.rotationMatrix();
                double time_stamp = element->time_stamp_;
                if(Config::Get<int>("set_saves_poses_kitti")){
                    outputFile << rotation_matrix(0, 0) << " " << rotation_matrix(0, 1) << " " << rotation_matrix(0, 2) << " " << translation.x() << " " 
                           << rotation_matrix(1, 0) << " " << rotation_matrix(1, 1) << " " << rotation_matrix(1, 2) << " " << translation.y() << " " 
                           << rotation_matrix(2, 0) << " " << rotation_matrix(2, 1) << " " << rotation_matrix(2, 2) << " " << translation.z() << "\n";
                }
                
                if(Config::Get<int>("set_saves_poses_tum")){
                    Eigen::Quaterniond quaternion(rotation_matrix);
                    outputFile << std::fixed << std::setprecision(8) << time_stamp << " " << translation.x() << " " << translation.y() << " " << translation.z() << " " 
                           << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << "\n";
                }

            } else {
                LOG(INFO) << "saves plotraject error";}
            }
            outputFile.close();
        }
        

        LOG(INFO) << "Average time tatol countfor_tracklast: " << frontend_->countfor_tracklast.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_tracklast: " << frontend_->countfor_tracklast.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_tracklast: " << frontend_->countfor_tracklast.max_frame_rate_ << " ms." << " \n ";

        LOG(INFO) << "Average time tatol countfor_estimate: " << frontend_->countfor_estimate.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_estimate: " << frontend_->countfor_estimate.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_estimate: " << frontend_->countfor_estimate.max_frame_rate_ << " ms." << " \n ";



        LOG(INFO) << "Average time tatol countfor_detect: " << frontend_->countfor_detect.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_detect: " << frontend_->countfor_detect.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_detect: " << frontend_->countfor_detect.max_frame_rate_ << " ms." << " \n ";

        LOG(INFO) << "Average time tatol countfor_findR: " << frontend_->countfor_findR.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_findR: " << frontend_->countfor_findR.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_findR: " << frontend_->countfor_findR.max_frame_rate_ << " ms." << " \n ";

        LOG(INFO) << "Average time tatol countfor_trian: " << frontend_->countfor_trian.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_trian: " << frontend_->countfor_trian.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_trian: " << frontend_->countfor_trian.max_frame_rate_ << " ms." << " \n ";

        LOG(INFO) << "Average time tatol countfor_back: " << frontend_->countfor_back_update.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_back: " << frontend_->countfor_back_update.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_back: " << frontend_->countfor_back_update.max_frame_rate_ << " ms." << " \n ";

        LOG(INFO) << "Average time tatol countfor_view: " << frontend_->countfor_view_update.GetAverageFrameRate() << " ms__.";
        LOG(INFO) << "mini time tatol countfor_view: " << frontend_->countfor_view_update.min_frame_rate_ << " ms.";
        LOG(INFO) << "max time tatol countfor_view: " << frontend_->countfor_view_update.max_frame_rate_ << " ms." << " \n ";


        float tatol = frontend_->countfor_tracklast.GetAverageFrameRate() + frontend_->countfor_estimate.GetAverageFrameRate()+
            frontend_->countfor_detect.GetAverageFrameRate() + frontend_->countfor_findR.GetAverageFrameRate() + frontend_->countfor_trian.GetAverageFrameRate()+
            frontend_->countfor_back_update.GetAverageFrameRate() + frontend_->countfor_view_update.GetAverageFrameRate();
            
        LOG(INFO) << "time tatol time count _add_all: " << tatol << " ms." << " \n ";

        LOG(INFO) << "mini frequency tatol: " << countfor_vo.min_frame_rate_ << " Hz." << " \n ";
        LOG(INFO) << "max frequency tatol: " << countfor_vo.max_frame_rate_ << " Hz." << " \n ";
        LOG(INFO) << "Average frequency tatol: " << countfor_vo.GetAverageFrameRate() << " Hz.";

        break;
            
        }
    }
    if(Config::Get<int>("if_viewer_open"))
    while(!flag_close_viewer){
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        flag_close_viewer = viewer_->GetCloseFlag();
    }
    if(Config::Get<int>("if_viewer_open"))
    viewer_->Close();
    LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    Frame::Ptr new_frame;
    auto t1 = std::chrono::high_resolution_clock::now();
    if(use_realworld_flag){
        new_frame = dataset_->RealworldNextFrame(); 
        if (new_frame == nullptr) {return true;}
    }else{
        new_frame = dataset_->DatasetNextFrame();  
        if (new_frame == nullptr) {return false;}
    }
    step_image_index_++;
    bool success = frontend_->AddFrame(new_frame);
    if(success == false) LOG(INFO) << "step_image_index: " << step_image_index_;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    // LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    countfor_vo.step_count_++;
    countfor_vo.total_time_ += time_used.count();

    // 每1秒输出一次平均频率
    if (countfor_vo.total_time_ >= 999.0) {
        double average_frequency = (countfor_vo.step_count_ / countfor_vo.total_time_)*1000;
        countfor_vo.RecordFrameRate(average_frequency);
        LOG(INFO) << "Average step frequency : " << average_frequency << " Hz. " << average_frequency ;
        countfor_vo.step_count_ = 0;
        countfor_vo.total_time_ = 0.0;
    }

 
    return success;
}



}  // namespace orangeslam
