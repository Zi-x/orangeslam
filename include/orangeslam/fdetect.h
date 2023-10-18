#ifndef ORANGESLAM_FDETECT_H_
#define ORANGESLAM_FDETECT_H_
#include "orangeslam/yolofastestv2.h" 
#include "orangeslam/common_include.h"
#include "orangeslam/frame.h"
namespace orangeslam{

enum class FdetectStatus { DETECT_NONE, DETECT_OK, DETECT_TOOBIG, DETECT_TOOSMALL };

class Fdetect{
    
    public:
     typedef std::shared_ptr<Fdetect> Ptr;

     Fdetect();

     void Close();

     bool fdetect_running = true;


     void DetectCurrentFrame(Frame::Ptr current_frame);

     std::mutex fdetect_completed_mutex_;

     bool fdetect_completed_bool_ = false;

     std::condition_variable fdetect_completed_condition_;

     std::vector<cv::Rect> fdetect_result;

     int fdetect_frame_num = 0;

    private:


     double fdetect_scale = 1.0;

     void Fdetect_Thread_Loop();

     std::thread Fdetet_thread_;

     Frame::Ptr current_frame_ = nullptr;

     FdetectStatus detect_status_ = FdetectStatus::DETECT_NONE;

     // mutex 
     std::mutex fdetect_frame_update_mutex_;

     std::condition_variable fdetect_frame_update_condition_;

     

     


};
}
#endif