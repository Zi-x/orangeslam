#include <opencv2/highgui/highgui_c.h>
#include <chrono>
#include "orangeslam/fdetect.h"
namespace orangeslam {

Fdetect::Fdetect() {  
    Fdetet_thread_ = std::thread(std::bind(&Fdetect::Fdetect_Thread_Loop, this));
}

void Fdetect::DetectCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lock(fdetect_frame_update_mutex_);
    current_frame_ = current_frame;
	fdetect_completed_bool_ = false;
	fdetect_frame_update_condition_.notify_one();

}

void Fdetect::Close() {
    fdetect_running = false; // 跳出循环
    Fdetet_thread_.join(); // 等待线程的结束
}

void Fdetect::Fdetect_Thread_Loop()
{

	static const char* class_name = "person";	
	yoloFastestv2 api;
	api.loadModel("../ncnn_model/yolo-fastestv2-opt.param",
		"../ncnn_model/yolo-fastestv2-opt.bin");

	std::vector<TargetBox> boxes;
 	
	
	int fdetect_frame_num = 0;
	cv::Mat frame;
	// fdetect_running
	while(fdetect_running){

		{
			std::unique_lock<std::mutex> lock(fdetect_frame_update_mutex_);
			LOG(INFO) << "wait lock in fdetect ";
			fdetect_frame_update_condition_.wait(lock); // 检查wait
			frame = current_frame_->left_img_.clone();			
		}

		auto start_time = std::chrono::high_resolution_clock::now();
		// frame = cv::imread("../ncnn_model/3.jpg");
		api.detection(frame, boxes);  // 耗时的操作
		auto end_time = std::chrono::high_resolution_clock::now();
		auto used_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time ).count();
		LOG(INFO) << "fdetect_person time used: "<< used_time << " ms";
		detect_status_ = FdetectStatus::DETECT_OK;
		if(boxes.size() == 0) detect_status_ = FdetectStatus::DETECT_NONE;
		// LOG(INFO) << "boxes.size(): "<< boxes.size();
		
		// FdetectStatus::DETECT_OK
		if(detect_status_ == FdetectStatus::DETECT_OK){
			
			for (int i = 0; i < int(boxes.size()); i++) {
				// std::cout << boxes[i].x1 << " " << boxes[i].y1 << " " << boxes[i].x2 << " " << boxes[i].y2
				// 	<< " " << boxes[i].score << " " << boxes[i].cate << std::endl;
				if(boxes[i].score * 100 <= 60) continue;
				char text[256];
				sprintf(text, "%s %.1f%%", class_name, boxes[i].score * 100);

				int baseLine = 0;
				cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				int x = boxes[i].x1;
				int y = boxes[i].y1 - label_size.height - baseLine;
				if (y < 0)
					y = 0;
				if (x + label_size.width > frame.cols)
					x = frame.cols - label_size.width;
				
				cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
					cv::Scalar(255, 255, 255), -1);
				cv::putText(frame, text, cv::Point(x, y + label_size.height),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

				cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
				cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
				
				
				fdetect_frame_num ++;
				// cv::rectangle
				// topLeft: 矩形的左上角顶点坐标。
				// bottomRight: 矩形的右下角顶点坐标。
				cv::Point point_ul(boxes[i].x1, boxes[i].y1);
				cv::Point point_br(boxes[i].x2, boxes[i].y2);
				fdetect_result.emplace_back(point_ul,point_br);
				LOG(INFO) << "point_ul: "<< point_ul;
				LOG(INFO) << "point_br: "<< point_br;
				
				
			}

			
			
			// cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
			// cv::imshow("img",frame);
			// cv::waitKey(1);

		}
		
		LOG(INFO) << "nofity fdetect_num: " << fdetect_frame_num;
		// 偶尔陷入wait，不知道是哪里，给fdetect_result加一个锁试试
		fdetect_completed_bool_ = true;
		fdetect_completed_condition_.notify_one();
		
	
	}

}
}