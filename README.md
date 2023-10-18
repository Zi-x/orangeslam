# orangeslam
本项目框架源于高翔博士著书《视觉SLAM14讲》中第十三讲的实践例<br>
硬件使用orangepi5，rk3588s，4大核A76，4小核A55<br>
整体系统结构没有大变化，但是前端frontend及可视化viewer做了较大改动，并增加了一个新的目标检测线程<br>
还改进了一些小功能<br>
轨迹保存<br>
&nbsp;&nbsp;&nbsp;·数据集跑完可自动保存轨迹（可选kitti或者tum格式）<br>
&nbsp;&nbsp;&nbsp;·增加真实相机入口，关闭可视化窗口时触发保存轨迹<br>
帧数计算<br>
&nbsp;&nbsp;&nbsp;·添加了一个帧数计算的类，能够输入每秒系统帧率<br>
&nbsp;&nbsp;&nbsp;&nbsp;同时当程序自然退出，能输出各步骤的最小/大用时、平均用时与帧率<br>
效率改进<br>
&nbsp;&nbsp;&nbsp;·增加关键帧更新viewer机制，可大大加快主线程帧率<br>
&nbsp;&nbsp;&nbsp;·改善后端backend中一处nullptr的bug（虽然很难触发，把后端阈值过小会触发coredump）<br>
&nbsp;&nbsp;&nbsp;·改善数据集跑完后viewer自动退出问题，便于查看最终轨迹<br>
自定义参数<br>
&nbsp;&nbsp;&nbsp;·在yaml增加了许多可自定义的参数，便于调试<br>
即使部署在orangepi5上也能够流畅运行<br>
若不开启目标检测线程，kitti数据集能达到60fps甚至100fps（不算图片读取时间），开启则有30fps以上<br>
对于现实世界，在读取camera（usb3.0）图片耗费大量时间的情况下，帧数也能在30fps上下（取决于双目分辨率，此处为720p*2560p）<br>
当然这是没有开启目标检测的情况下，开启目标检测若要保持实时性能需要降低分辨率，最好是降低camera源头输出分辨率（内参就是需要解决的问题了）<br>
综上，我就厚颜无耻的将项目命名为orangeslam （~生活所迫，高博见谅~）<br>
好了<br>
剩下项目介绍及实验图片视频正在施工中...
