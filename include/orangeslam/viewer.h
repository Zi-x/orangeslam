#ifndef ORANGESLAM_VIEWER_H
#define ORANGESLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "orangeslam/common_include.h"
#include "orangeslam/frame.h"
#include "orangeslam/map.h"

namespace orangeslam {

/**
 * 可视化
 */
class Viewer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    void SetMap(Map::Ptr map) { map_ = map; }

    void Close();

    // 增加一个当前帧
    void AddCurrentFrame(Frame::Ptr current_frame);

    // 更新地图
    void UpdateMap();

    void SetCloseFlag(bool flag){close_viewer_flag = flag;}

    bool GetCloseFlag(){return close_viewer_flag;}

   private:
    void ThreadLoop();

    void DrawFrame(Frame::Ptr frame, const float* color);

    void Draw2dTrajectory();

    void DrawMapPoints();

    void DrawAxis(const float size = 1.0f);

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    void Follow2dTrajectory(pangolin::OpenGlRenderState& vis_camera);

    void Draw2dMapPoints();

    void Draw2dCurrentFrame(Frame::Ptr frame, const float* color);

    void SavesTrajectory();

    float adaptiveMap(float value, float minVal, float maxVal);

    std::array<float, 3> getColor(float value);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    bool close_viewer_flag = false;

    std::unordered_map<unsigned long, Frame::Ptr> all_keyframes_;
    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
    std::unordered_map<unsigned long, MapPoint::Ptr> all_landmarks_;

    bool map_updated_ = false;

    std::mutex viewer_data_mutex_;

    std::condition_variable viewer_map_update_;
};
}  // namespace orangeslam

#endif  // ORANGESLAM_VIEWER_H
