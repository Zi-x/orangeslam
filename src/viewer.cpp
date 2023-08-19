
#include "orangeslam/viewer.h"
#include "orangeslam/feature.h"
#include "orangeslam/frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace orangeslam {

Viewer::Viewer() {
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
}

void Viewer::Close() {
    viewer_running_ = false; // 跳出循环
    viewer_thread_.join(); // 等待线程的结束
}

// 前端每次pose估计都会执行AddCurrentFrame
// 而all_keyframes_只会是关键帧才执行，所以all_keyframes_是滞后的
void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    current_frame_ = current_frame;
}

void Viewer::UpdateMap() {
    assert(map_ != nullptr);

    std::unique_lock<std::mutex> lock(viewer_data_mutex_);
    viewer_map_update_.notify_one();
    
    // std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    // all_keyframes_ = map_->GetAllKeyFrames();
    // active_keyframes_ = map_->GetActiveKeyFrames();
    // all_landmarks_ = map_->GetAllMapPoints();
    // active_landmarks_ = map_->GetActiveMapPoints();
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    int WIN_WIDTH = 1137;
    int WIN_HEIGHT = 640;
    int UI_WIDTH = 100;
    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};
    const float red[3] = {1, 0, 0};
    
    pangolin::CreateWindowAndBind("OrangeSLAM", WIN_WIDTH, WIN_HEIGHT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ModelViewLookAt
    // eyeX、eyeY和eyeZ表示相机的位置坐标，即相机的眼睛位置
    // 参数centerX、centerY和centerZ表示相机观察的目标点坐标，即相机的注视点
    // 参数upX、upY和upZ表示相机的上方向向量
    pangolin::OpenGlRenderState vis_2d_camera(
        pangolin::ProjectionMatrix(WIN_WIDTH, WIN_HEIGHT, 400, 400, 512, 384, 0.1, 1500),
        pangolin::ModelViewLookAt(0, -100, 40, 0, 0, 50, 0, -1.0, 0.0));    


    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(WIN_WIDTH, WIN_HEIGHT, 400, 400, 512, 384, 0.1, 1500),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));
    
    // Add named OpenGL viewport to window and provide 3D Handler
    
    // 0.0, 1.0：视口在窗口中的垂直范围，从下到上的比例
    // 0.0, 1.0：视口在窗口中的水平范围，从左到右的比例。
    pangolin::View& vis_2d_display =
        pangolin::Display("2d-trajectory")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.7, 768.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_2d_camera));

    
    pangolin::View& vis_display =
        pangolin::Display("3d-Frame")
            .SetBounds(0.5, 1.0,0.5, 1.0, 4.0f / 4.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera))
            .SetLock(pangolin::LockRight, pangolin::LockTop);

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    // 创建控制面板的控件对象，pangolin中

    pangolin::Var<std::function<void()>> saveTra( "ui.SaveTra", [this]() {
    SavesTrajectory();
});

    while (!pangolin::ShouldQuit() && viewer_running_) {

        // if(map_updated_){
        //     std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        //     all_keyframes_ = map_->GetAllKeyFrames();
        //     active_landmarks_ = map_->GetActiveMapPoints();
        //     map_updated_ = false;

        // }

        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        viewer_map_update_.wait(lock);
        all_keyframes_ = map_->GetAllKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_2d_display.Activate(vis_2d_camera);
        // std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        // std::unique_lock<std::mutex> lock(viewer_data_mutex_); 关掉，找出偶发性segmentation fault (core dumped)，出现原因--算了，找不出问题
        if(current_frame_){ //用flag出问题,是不是因为这里要加锁？
            Draw2dCurrentFrame(current_frame_, green);
            cv::Mat img = PlotFrameImage();
            cv::imshow("image", img);
            cv::waitKey(1);
        }
        if (all_keyframes_.size() > 0) {
            DrawAxis(5.0f); 
            Draw2dMapPoints();
            Draw2dTrajectory();
            Follow2dTrajectory(vis_2d_camera);
        }

        // vis_display.Activate(vis_camera);

        // if (current_frame_) {
        //  // 绿色相机，画当前帧，和特征点
            
        //  DrawFrame(current_frame_, green);
        //  FollowCurrentFrame(vis_camera);
            
        // }
        // if (map_) {
        //     DrawMapPoints();
        // }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); /// now use notify and wait
    }
    // SavesTrajectory();
    SetCloseFlag(true);
    LOG(INFO) << "Stop viewer";
}

cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out;
    cv::cvtColor(current_frame_->left_img_, img_out, CV_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.lock()) {
            auto feat = current_frame_->features_left_[i];
            cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0),
                       2);
        }
    }
    return img_out;
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    SE3 Twc = current_frame_->Pose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
    
}

void Viewer::Follow2dTrajectory(pangolin::OpenGlRenderState& vis_camera) {
    
    SE3 Twc = current_frame_->Pose().inverse();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Twc.setRotationMatrix(rotation);
    
    Eigen::Vector3d translation_copy = Twc.translation();
    translation_copy[1] = 0;
    Twc.translation() = translation_copy;


    //std::cout << "SE3:\n" << Twc.matrix() << std::endl;
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    SE3 Twc = frame->Pose().inverse();
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();
    // 函数将转换后的矩阵与当前的模型视图矩阵相乘
    // 将原有的世界坐标系变换到当前坐标系下，将当前帧设为坐标原点，便于直接在原点画出相机模型
    // SE3是double型glMultMatrixf是需要float型
    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat*)m.data());

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::DrawMapPoints() {
    const float red[3] = {1.0, 0, 0};
    for (auto& kf : active_keyframes_) {
        // 红色相机，画过去帧
        DrawFrame(kf.second, red);
    }

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto& landmark : active_landmarks_) {
        auto pos = landmark.second->Pos();
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

void Viewer::Draw2dMapPoints() {
    const float red[3] = {1, 0.6, 0.13};
    glPointSize(0.5);
    glBegin(GL_POINTS);
    for (auto& landmark : active_landmarks_) {
        auto pos = landmark.second->Pos();
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(pos[0], 0, pos[2]);
    }
    glEnd();
}

void Viewer::Draw2dCurrentFrame(Frame::Ptr frame, const float* color){

    if (color == nullptr) {
        glColor3f(0, 1, 0);
    } else
        glColor3f(color[0], color[1], color[2]);
    SE3 Twc = frame->Pose().inverse();
    Eigen::Vector3d translation_copy = Twc.translation();
    translation_copy[1] = 0;
    Twc.translation() = translation_copy;

    // translation = Twc.translation();
    // glPointSize(2);
    // glVertex3d(translation[0], 0, translation[2]);
    // glEnd();

    // 函数将转换后的矩阵与当前的模型视图矩阵相乘
    // 将原有的世界坐标系变换到当前坐标系下，将当前帧设为坐标原点
    glPushMatrix();
    // template cast<float>() 是一个模板转换操作，将矩阵中的元素类型转换为float类型
    // template关键字用于告诉编译器，这是一个模板函数或成员函数
    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat*)m.data());
    DrawAxis(3.0f);
    glPointSize(7);
    glBegin(GL_POINTS);
    glVertex3d(0,0,0);
    glEnd();
    glPopMatrix();
    
}

void Viewer::Draw2dTrajectory(){
    float minVal = -100.0;
    float maxVal = 100.0; 
    for (size_t i = 0; i < all_keyframes_.size() -1; i++) {
        Eigen::Vector3d translation1(0.0, 0.0, 0.0), translation2(0.0, 0.0, 0.0);
        auto it = all_keyframes_.find(i);
        if (it != all_keyframes_.end()) {
            auto element = it->second;
            SE3 Twc = element->Pose().inverse();
            translation1 = Twc.translation();
        } else {
            std::cout << "error Draw2dTrajectory i 元素不存在" << std::endl;
        }
        it = all_keyframes_.find(i+1);
        if (it != all_keyframes_.end()) {
            auto element = it->second;
            SE3 Twc = element->Pose().inverse();
            translation2 = Twc.translation();   
        } else {
            std::cout << "error Draw2dTrajectory i+1 元素不存在" << std::endl;
        }
        float high = (translation1.y() + translation2.y())/2;

        
        float colormap = adaptiveMap(high, minVal, maxVal);
        
        glColor3f(getColor(colormap)[0],getColor(colormap)[1],getColor(colormap)[2]);
        glLineWidth(1);
        glBegin(GL_LINES);
        glVertex3d(translation1.x(), 0, translation1.z());
        glVertex3d(translation2.x(), 0, translation2.z());
        
        glEnd();
        // glBegin(GL_POINTS);
        // glVertex3d(translation1.x(), 0, translation1.z());
        // glVertex3d(translation2.x(), 0, translation2.z());
        // glEnd();
        //std::cout << translation1.y() << std::endl;
        
    }

}

void Viewer::SavesTrajectory(){
    std::ofstream outFile("pose.txt"); // 创建输出文件流
    auto t1 = std::chrono::steady_clock::now();
    
    if (outFile.is_open()) {
        for (size_t i = 0; i < all_keyframes_.size(); i++) {
            Eigen::Vector3d translation(0.0, 0.0, 0.0);
            auto it = all_keyframes_.find(i);
            if (it != all_keyframes_.end()) {
                auto element = it->second;
                SE3 Twc = element->Pose().inverse();
                translation = Twc.translation();
                outFile << i << " ";
                for(int j = 0; j < 3; j++){
                    outFile << translation[j] << " ";
                }
            } else {
                std::cout << "error SavesTrajectory i 元素不存在" << std::endl;
            }
            outFile << std::endl;
            
        }
        outFile.close(); // 关闭文件流
    }
    else {
        std::cout << "Unable to open file!" << std::endl;
    }

    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "SavesTrajectory cost time: " << time_used.count() << " seconds.";

}

float Viewer::adaptiveMap(float value, float minVal, float maxVal) {
    // 将值限制在最小值和最大值之间
    float clampedValue = std::max(minVal, std::min(maxVal, value));
    // 线性映射到0到1的范围
    float mappedValue = (clampedValue - minVal) / (maxVal - minVal);
    return mappedValue;
}

std::array<float, 3> Viewer::getColor(float value) {
    float b = 1 - value;    // 设置红色分量，值越小红色越高
    float r = value;        // 设置蓝色分量，值越大蓝色越高
    float g = 0;            // 设置绿色分量为0
    return {r, g, b};
}


void Viewer::DrawAxis(const float size){
    // 绘制 X 轴
    glColor3f(1.0f, 0.0f, 0.0f); // 设置颜色为红色 X 轴
    glLineWidth(2.0f); 
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(size, 0.0f, 0.0f);
    glEnd();

    // 绘制 Y 轴
    glColor3f(0.0f, size, 0.0f); // 设置颜色为绿色 Y 轴
    glLineWidth(2.0f); 
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, size, 0.0f);
    glEnd();

    // 绘制 Z 轴
    glColor3f(0.0f, 0.0f, size); // 设置颜色为蓝色 Z 轴
    glLineWidth(2.0f); 
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, size);
    glEnd();
}

}  // namespace orangeslam
