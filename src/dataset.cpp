#include "orangeslam/dataset.h"
#include "orangeslam/frame.h"
#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;

namespace orangeslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    resize_scale = Config::Get<float>("resize_scale");
    if(Config::Get<int>("use_kitti")){
        // read camera intrinsics and extrinsics in dataset_kitti
        ifstream fin(dataset_path_ + "/calib.txt");
        if (!fin) {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            char camera_name[3];
            for (int k = 0; k < 3; ++k) {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k) {
                fin >> projection_data[k];
            }
            Mat33 K;
            // 607.2 185.2, 1214.4 370.4 -- 1241  376
            // P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 
            //     0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 
            //     0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
            // P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 
            //     0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 
            //     0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            Vec3 t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            LOG(INFO) << "Camera " << i << " intrinsics: \n" << K;

            K = K * resize_scale;
            Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                            t.norm(), SE3(SO3(), t)));
            cameras_.push_back(new_camera);
            LOG(INFO) << "Camera " << i << " extrinsics t: " << t.transpose();
        }
        fin.close();


        // 读取时间戳 -- 为了使用evo评估新增读取时间戳，转为tum格式
        std::ifstream inputFile(dataset_path_ + "/times.txt"); // 打开输入文件流

        if (!inputFile.is_open()) {
            std::cerr << "Failed to open the times txt." << std::endl;
            return 1; 
        }

        
        std::string line;
        while (std::getline(inputFile, line)) {
            // 将读取的行（字符串）转换为 double 类型的数字
            double value = std::stod(line);
            time_stamp_values.push_back(value); // 将值添加到向量中
        }

        inputFile.close(); // 关闭文件流
        return true;

    } else if(Config::Get<int>("use_openloris")){
        std::ifstream inputFile(dataset_path_ + "/fisheye1.txt"); // 打开输入文件流
        std::cerr << dataset_path_ + "/fisheye1.txt" << std::endl;
        if (inputFile.is_open()) {
            double number;
            std::string filename;
            int i = 0;
            while (inputFile >> number >> filename) {
                i++;
                if(i < Config::Get<int>("openloris_begin_frame")) continue;
                time_stamp_values_left.push_back(number);
                openloris_filenames_left.push_back(filename);
            }
            inputFile.close(); // 关闭文件
        } else {
            std::cerr << "无法打开文件 fisheye1" << std::endl;
            return 1;
        }

        std::ifstream inputFile2(dataset_path_ + "/fisheye2.txt"); // 打开输入文件流
        if (inputFile2.is_open()) {
            double number;
            std::string filename;
            int i = 0;
            while (inputFile2 >> number >> filename) {
                i++;
                if(i < Config::Get<int>("openloris_begin_frame")) continue;
                time_stamp_values_right.push_back(number);
                openloris_filenames_right.push_back(filename);
            }
            inputFile2.close(); // 关闭文件
        } else {
            std::cerr << "无法打开文件 fisheye2" << std::endl;
            return 1;
        }
        
        for(int i = 0; i<2; i++){
            std::string param_name1;
            std::string param_name2;
            if(Config::Get<std::string>("use_openloris_scene") == "corridor1"){
                param_name1 = "corridor1.t265_fisheye" + std::to_string(i+1) + "_optical_frame.intrinsics";
                param_name2 = "corridor1.t265_fisheye" + std::to_string(i+1) + "_optical_frame.distortion_coefficients";
            }else if(Config::Get<std::string>("use_openloris_scene") == "market1"){
                param_name1 = "market1.t265_fisheye" + std::to_string(i+1) + "_optical_frame.intrinsics";
                param_name2 = "market1.t265_fisheye" + std::to_string(i+1) + "_optical_frame.distortion_coefficients";
            }
            
            cv::Mat intrinsics = Config::Get<cv::Mat>(param_name1);
            cv::Mat distortion = Config::Get<cv::Mat>(param_name2);

            if(i == 0){
                openloris_cameraMatrix_left = (cv::Mat_<double>(3, 3) << intrinsics.at<double>(0, 0), 0, intrinsics.at<double>(0, 1),
                                                0,intrinsics.at<double>(0, 2), intrinsics.at<double>(0, 3),
                                                0, 0, 1);
                Vec3 t(0,0,0);
                cv::Mat K = openloris_cameraMatrix_left * resize_scale;
                distortionCoefficients_left = (cv::Mat_<double>(1, 4) << distortion.at<double>(0, 0), distortion.at<double>(0, 1), distortion.at<double>(0, 2),
                                            distortion.at<double>(0, 3));
                // Camera(double fx, double fy, double cx, double cy, double baseline,const SE3 &pose)
                Camera::Ptr new_camera(new Camera(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
                                                t.norm(), SE3(SO3(), t)));
                // std::cout << distortionCoefficients_left << std::endl;
                // std::cout << openloris_cameraMatrix_left << std::endl;
                cameras_.push_back(new_camera);
                // std::cout << "K: " << K << std::endl;

                }else if(i == 1){
                openloris_cameraMatrix_right = (cv::Mat_<double>(3, 3) << intrinsics.at<double>(0, 0), 0, intrinsics.at<double>(0, 1),
                                                0,intrinsics.at<double>(0, 2), intrinsics.at<double>(0, 3),
                                                0, 0, 1);
                distortionCoefficients_right = (cv::Mat_<double>(1, 4) << distortion.at<double>(0, 0), distortion.at<double>(0, 1), distortion.at<double>(0, 2),
                                            distortion.at<double>(0, 3));
                cv::Mat transMat;
                if(Config::Get<std::string>("use_openloris_scene") == "corridor1"){
                    transMat = Config::Get<cv::Mat>("corridor1.trans_matrix.matrix_fisheye1_to_fisheye2");
                }else if(Config::Get<std::string>("use_openloris_scene") == "market1"){
                    transMat = Config::Get<cv::Mat>("market1.trans_matrix.matrix_fisheye1_to_fisheye2");
                }
                
                Eigen::Matrix4d eigentransMat;
                for(int i=0;i<4;i++)
                    for(int j=0;j<4;j++)
                        eigentransMat(i,j) = transMat.at<double>(i,j);  

                // std::cout << "eigentransMat: " << eigentransMat << std::endl;
                
                // std::cin.get();

                T_openloris = SE3(eigentransMat);
                Eigen::Quaterniond quaternion(T_openloris.inverse().rotationMatrix());
                double roll, pitch, yaw;
                roll = atan2(2 * (quaternion.w() * quaternion.x() + quaternion.y() * quaternion.z()),
                            1 - 2 * (quaternion.x() * quaternion.x() + quaternion.y() * quaternion.y()));
                pitch = asin(2 * (quaternion.w() * quaternion.y() - quaternion.z() * quaternion.x()));
                yaw = atan2(2 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y()),
                            1 - 2 * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z()));
                // Print Euler angles
                // Roll: -0.215388 d
                // Pitch: -0.398649 d
                // Yaw: -0.176814 d
                std::cout << "Roll: " << roll * (180.0 / M_PI) << " d" << std::endl;
                std::cout << "Pitch: " << pitch * (180.0 / M_PI) << " d" << std::endl;
                std::cout << "Yaw: " << yaw * (180.0 / M_PI) << " d" << std::endl;
                cv::Mat K = openloris_cameraMatrix_right * resize_scale;
        
                Camera::Ptr new_camera(new Camera(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
                                                 T_openloris.translation().norm(), T_openloris.inverse())); //    SE3(SO3(), vector3d))
                cameras_.push_back(new_camera);
                // T.translation():   -0.0639779  5.06667e-05 -4.61147e-05
                std::cout << "T.translation(): " << T_openloris.inverse().translation().transpose() << std::endl;
            }
            
        }      
        return true;

    }else if(Config::Get<int>("use_realworld")){
        for(int i = 0; i<2; i++){
            std::string param_name1 = "realworld.stereo" + std::to_string(i+1) + ".intrinsics";
            std::string param_name2 = "realworld.stereo" + std::to_string(i+1) + ".distortion_coefficients";
            cv::Mat intrinsics = Config::Get<cv::Mat>(param_name1);
            cv::Mat distortion = Config::Get<cv::Mat>(param_name2);

            if(i == 0){
                realworld_cameraMatrix_left = (cv::Mat_<double>(3, 3) << intrinsics.at<double>(0, 0), 0, intrinsics.at<double>(0, 2),
                                                0,intrinsics.at<double>(1, 1), intrinsics.at<double>(1, 2),
                                                0, 0, 1);
                Vec3 t(0,0,0);
                // K乘以了resize
                cv::Mat K = realworld_cameraMatrix_left * resize_scale;
                distortionCoefficients_left = (cv::Mat_<double>(1, 4) << distortion.at<double>(0, 0), distortion.at<double>(0, 1), 0, 0);
                // Camera(double fx, double fy, double cx, double cy, double baseline,const SE3 &pose)
                Camera::Ptr new_camera(new Camera(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
                                                t.norm(), SE3(SO3(), t)));
                
                cameras_.push_back(new_camera);
                // std::cout << "K: " << K << std::endl;
                }else if(i == 1){
                realworld_cameraMatrix_right = (cv::Mat_<double>(3, 3) << intrinsics.at<double>(0, 0), 0, intrinsics.at<double>(0, 2),
                                                0,intrinsics.at<double>(1, 1), intrinsics.at<double>(1, 2),
                                                0, 0, 1);
                distortionCoefficients_right = (cv::Mat_<double>(1, 4) << distortion.at<double>(0, 0), distortion.at<double>(0, 1), 0, 0);
                cv::Mat transMat = Config::Get<cv::Mat>("realworld.trans_matrix.matrix_l_to_r");
                
                Eigen::Matrix4d eigentransMat;
                for(int i=0;i<4;i++)
                    for(int j=0;j<4;j++)
                        eigentransMat(i,j) = transMat.at<double>(i,j);  

                // std::cout << "eigentransMat: " << eigentransMat << std::endl;

                T_realworld = SE3(eigentransMat);
                Eigen::Quaterniond quaternion(T_realworld.rotationMatrix());
                double roll, pitch, yaw;
                roll = atan2(2 * (quaternion.w() * quaternion.x() + quaternion.y() * quaternion.z()),
                            1 - 2 * (quaternion.x() * quaternion.x() + quaternion.y() * quaternion.y()));
                pitch = asin(2 * (quaternion.w() * quaternion.y() - quaternion.z() * quaternion.x()));
                yaw = atan2(2 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y()),
                            1 - 2 * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z()));

                std::cout << "Roll: " << roll * (180.0 / M_PI) << " d" << std::endl;
                std::cout << "Pitch: " << pitch * (180.0 / M_PI) << " d" << std::endl;
                std::cout << "Yaw: " << yaw * (180.0 / M_PI) << " d" << std::endl;
                // K乘以了resize
                cv::Mat K = realworld_cameraMatrix_right * resize_scale;
                // Eigen::Vector3d vector3d(T.inverse().translation()[0],0,0);
                Camera::Ptr new_camera(new Camera(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
                                                 T_realworld.translation().norm(), T_realworld)); //    SE3(SO3(), vector3d))
                cameras_.push_back(new_camera);
                std::cout << "T.translation(): " << T_realworld.translation().transpose() << std::endl;
                
            }
            // std::cin.get();
            
        }   
        return true;
    }
    return false;


}
bool Dataset::initRealworldCamera(){
    std::cout  << "fps" << std::endl;
    if (!cap.open(0, cv::CAP_V4L)) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return false;
    }
    cap.set(cv::CAP_PROP_FPS, 60);  // 帧率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 2560);   
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720); 
    // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));  // 视频流格式
    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));

    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    std::cout << "FourCC: " << char(fourcc & 255) << char((fourcc >> 8) & 255)
              << char((fourcc >> 16) & 255) << char((fourcc >> 24) & 255) << std::endl;
    double fps = cap.get(cv::CAP_PROP_FPS);
    

    std::cout << "帧率: " << fps << std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(2101));
    return true;
}

Frame::Ptr Dataset::RealworldNextFrame() {
	cv::Mat frame;
    auto before = std::chrono::high_resolution_clock::now();
    try {
        cap >> frame;
    } catch (const std::exception& e) {
       std::cerr << "发生异常: " << e.what() << std::endl;
       return nullptr;
    }
    
    // if (frame.empty()) {
    //     std::cerr << "Error: Camera Frame is not empty." << std::endl;
    //     return nullptr;
    // } 
    // if (leftImageRec.data == nullptr || rightImageRec.data == nullptr) {
    //     LOG(WARNING) << "cannot find images at index " << current_image_index_;
    //     return nullptr;
    // }

    auto after=  std::chrono::high_resolution_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
    LOG(INFO) << "camera cap cost time: " << time_used.count() << " ms.";

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat leftImageRec = frame(cv::Rect(0, 0, 1280, 720));
    cv::Mat rightImageRec = frame(cv::Rect(1280, 0, 1280, 720));

    // cv::imshow("Camera Frame l", leftImageRec);
    // cv::imshow("Camera Frame r", rightImageRec);
    // cv::waitKey(0);
    // 縮小圖像，暫時關閉，resize 爲1
    // cv::Mat image_left_resized, image_right_resized;
    // cv::resize(leftImageRec, image_left_resized, cv::Size(), resize_scale, resize_scale,
    //         cv::INTER_NEAREST);
    // cv::resize(rightImageRec, image_right_resized, cv::Size(), resize_scale, resize_scale,
    //         cv::INTER_NEAREST);
    // auto after_2= std::chrono::high_resolution_clock::now();
    // auto time_used2 = std::chrono::duration_cast<std::chrono::milliseconds>(after_2 - after);
    // LOG(INFO) << "image resize and process cost time: " << time_used2.count() << " ms.";
    // 去畸變
    // if(current_image_index_ == 0){
    //     cv::Size size_left(leftImageRec.cols, leftImageRec.rows);
    //     cv::Size size_right(rightImageRec.cols, rightImageRec.rows);

    //     cv::Mat Rl, Rr, Pl, Pr, Q;
    //     cv::Mat cvMat_r(3, 3, CV_64F);
    //     cv::Mat cvMat_t(3, 1, CV_64F);
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //             cvMat_r.at<double>(i, j) = T_realworld.rotationMatrix()(i, j);
    //         }
    //     }
    //     for (int i = 0; i < 3; ++i) {
    //             cvMat_t.at<double>(i, 0) = T_realworld.translation()(i, 0);
    //     }

    //     cout << "cr: \n" << endl << cvMat_r << endl;
    //     cout << "ct: \n" << endl << cvMat_t << endl;
    //     // 雙目矯正，應該要改完糾正後的左右目R和內參p，如有必要也要改t，暫時不想
    //     cv::stereoRectify(realworld_cameraMatrix_left, distortionCoefficients_left, realworld_cameraMatrix_right, distortionCoefficients_right, size_left,
    //         cvMat_r, cvMat_t, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 0);
    //     cout << "Rl: " << endl << Rl << endl;
    //     cout << "Rr: " << endl << Rr << endl;
    //     cout << "pl: " << endl << Pl << endl;
    //     cout << "pr: " << endl << Pr << endl;
    //     std::vector<double> vec1 = {762.098,762.098,577.96,390.6405};
    //     cameras_[1]->set_pose_(T_realworldtest);
    //     cameras_[0]->set_fx_fy_cx_cy_(vec1[0],vec1[1], vec1[2], vec1[3]);
    //     cameras_[1]->set_fx_fy_cx_cy_(vec1[0],vec1[1], vec1[2], vec1[3]);
    //     cv::initUndistortRectifyMap(realworld_cameraMatrix_left, distortionCoefficients_left, Rl, Pl.rowRange(0, 3).colRange(0, 3), size_left, CV_16SC2, map1_left, map2_left);
    //     cv::initUndistortRectifyMap(realworld_cameraMatrix_right, distortionCoefficients_right, Rr, Pr.rowRange(0, 3).colRange(0, 3), size_right, CV_16SC2, map1_right, map2_right);
        
        // cv::initUndistortRectifyMap(realworld_cameraMatrix_left, distortionCoefficients_left, cv::Mat(), realworld_cameraMatrix_left,
        // size_left, CV_16SC2, map1_left, map2_left);
        // cv::initUndistortRectifyMap(realworld_cameraMatrix_right, distortionCoefficients_right, cv::Mat(), realworld_cameraMatrix_right,
        // size_right, CV_16SC2, map1_right, map2_right); // CV_32FC1  CV_16SC2
            
        // }
    // cv::Mat img_rtf_l, img_rtf_r;

    // cv::remap(leftImageRec, leftImageRec, map1_left, map2_left, cv::INTER_LINEAR); 
    // cv::remap(rightImageRec, rightImageRec, map1_right, map2_right, cv::INTER_LINEAR);
    // cv::Mat img_src, img_rtf;
    // cv::hconcat(leftImageRec, rightImageRec, img_src);
    // cv::hconcat(img_rtf_l, img_rtf_r, img_rtf);
    

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = leftImageRec;
    new_frame->right_img_ = rightImageRec;
    
    // new_frame->time_stamp_ = time_stamp_values[current_image_index_];  realworld暫時不記錄時間，沒有真實軌跡，頂多到時候用apps記錄下軌跡，手動對其就行
    current_image_index_++;
    return new_frame;
}


Frame::Ptr Dataset::DatasetNextFrame() {
    if(Config::Get<int>("use_kitti")){
        boost::format fmt("%s/image_%d/%06d.png");
        cv::Mat image_left, image_right;
        // read images
        auto before = std::chrono::high_resolution_clock::now();

        image_left =
            cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                    cv::IMREAD_GRAYSCALE);
        image_right =
            cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                    cv::IMREAD_GRAYSCALE);

        if (image_left.data == nullptr || image_right.data == nullptr) {
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }
        
        auto after=  std::chrono::high_resolution_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
        LOG(INFO) << "image imread cost time: " << time_used.count() << " ms.";

        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), resize_scale, resize_scale,
                cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), resize_scale, resize_scale,
                cv::INTER_NEAREST);

        auto after_2= std::chrono::high_resolution_clock::now();
        auto time_used2 = std::chrono::duration_cast<std::chrono::milliseconds>(after_2 - after);

        LOG(INFO) << "image resize cost time: " << time_used2.count() << " ms.";
        
        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        
        new_frame->time_stamp_ = time_stamp_values[current_image_index_];
        current_image_index_++;
        return new_frame;

    } else if(Config::Get<int>("use_openloris")){
        // while(time_stamp_values_left[current_image_index_] != time_stamp_values_right[current_image_index_]) current_image_index_++;
        cv::Mat image_left, image_right;
        boost::format fmt("%s/%s");
        
        image_left =
            cv::imread((fmt % dataset_path_ % openloris_filenames_left[current_image_index_]).str(),
                    cv::IMREAD_GRAYSCALE);
        image_right =
            cv::imread((fmt % dataset_path_ % openloris_filenames_right[current_image_index_]).str(),
                    cv::IMREAD_GRAYSCALE);

        if (image_left.data == nullptr || image_right.data == nullptr) {
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }

        if(current_image_index_ == 0){
            cv::Size size_left(image_left.cols, image_left.rows);
            cv::Size size_right(image_right.cols, image_right.rows);

            // cv::Mat Rl, Rr, Pl, Pr, Q;
            // cv::Mat cvMat_r(3, 3, CV_64FC1, T_openloris.inverse().rotationMatrix().data());
            // cv::Mat cvMat_r(3, 3, CV_64F);
            // cv::Mat cvMat_t(3, 1, CV_64F);
            // for (int i = 0; i < 3; ++i) {
            //     for (int j = 0; j < 3; ++j) {
            //         cvMat_r.at<double>(i, j) = T_openloris.rotationMatrix()(i, j);
            //     }
            // }
            // for (int i = 0; i < 3; ++i) {
            //         cvMat_t.at<double>(i, 0) = T_openloris.inverse().translation()(i, 0);
            // }

            // cout << "cr: \n" << endl << cvMat_r << endl;
            // cout << "ct: \n" << endl << cvMat_t << endl;

            // cv::fisheye::stereoRectify(openloris_cameraMatrix_left, distortionCoefficients_left, openloris_cameraMatrix_right, distortionCoefficients_right, size_left,
            //     cvMat_r, cvMat_t, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, size_left);
            // cout << "Rl: " << endl << Rl << endl;
            // cout << "Rr: " << endl << Rr << endl;
            // cout << "pl: " << endl << Pl << endl;
            // cout << "pr: " << endl << Pr << endl;

            // cv::fisheye::initUndistortRectifyMap(openloris_cameraMatrix_left, distortionCoefficients_left, Rl, Pl.rowRange(0, 3).colRange(0, 3), size_left, CV_32F, map1_left, map2_left);
            // cv::fisheye::initUndistortRectifyMap(openloris_cameraMatrix_right, distortionCoefficients_right, Rr, Pr.rowRange(0, 3).colRange(0, 3), size_right, CV_32F, map1_right, map2_right);

            cv::fisheye::initUndistortRectifyMap(openloris_cameraMatrix_left, distortionCoefficients_left, cv::Mat(), openloris_cameraMatrix_left,
            size_left, CV_32F, map1_left, map2_left);
            cv::fisheye::initUndistortRectifyMap(openloris_cameraMatrix_right, distortionCoefficients_right, cv::Mat(), openloris_cameraMatrix_right,
            size_right, CV_32F, map1_right, map2_right); // CV_32FC1  CV_16SC2
            
        }
        // INTER_LINEAR   INTER_NEAREST
        // cv::Mat undistort_left, undistort_right;
        
        cv::remap(image_left, image_left, map1_left, map2_left, cv::INTER_LINEAR); 
        cv::remap(image_right, image_right, map1_right, map2_right, cv::INTER_LINEAR);/// CV_HAL_BORDER_CONSTANT BORDER_REPLICATE
    
        // cv::imshow("Left Image", image_left);
        // cv::Mat image_left_resized, image_right_resized;
        // cv::resize(image_left, image_left_resized, cv::Size(), resize_scale, resize_scale,
        //         cv::INTER_LINEAR);
        // cv::resize(image_right, image_right_resized, cv::Size(), resize_scale, resize_scale,
        //         cv::INTER_LINEAR);

        

        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left;
        new_frame->right_img_ = image_right;
        new_frame->time_stamp_ = time_stamp_values_left[current_image_index_];
        // if(current_image_index_<=3000)
        current_image_index_++;
        return new_frame;

    }
    return 0;
   
}

}  // namespace orangeslam