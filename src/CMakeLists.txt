# 创建一个名为 "orangeslam" 的共享库,这是一种可重用的代码模块，它包含了一组源代码文件，可以在多个程序中进行共享和链接
add_library(orangeslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp
        fdetect.cpp
        yolofastestv2.cpp)

target_link_libraries(orangeslam
        ${THIRD_PARTY_LIBS})
