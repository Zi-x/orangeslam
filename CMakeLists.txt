cmake_minimum_required(VERSION 2.8)
project(orangeslam)
set(CMAKE_BUILD_TYPE Release) #Release
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
# -std=c++14：指定编译器使用C++14标准进行编译。
# -O3：启用最高级别的优化。这将对代码进行各种优化，以获得更高的执行速度。
# -fopenmp：启用OpenMP支持，以便在代码中使用OpenMP多线程并行编程。
# -pthread：启用对POSIX线程库的支持，以便在代码中使用多线程。
# set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS "-std=gnu++14 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=gnu++14 -O3 -fopenmp -pthread")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g -O3")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
# set(CMAKE_PREFIX_PATH "/home/orangepi/opencv4") 
find_package(OpenCV 3 REQUIRED) 
# include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED 0.8)
include_directories(${Pangolin_INCLUDE_DIRS})
include(CTestUseLaunchers)
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})


# gflags
find_package(GFlags REQUIRED)
#include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# add_library(ncnn STATIC IMPORTED)
# set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/ncnnlib/lib/libncnn.a)

#THIRD_PARTY_LIBS 在子目录中进行target_link_libraries了
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES}  GL GLU GLEW glut #GLESv2 EGL
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        Sophus::Sophus
        ${PROJECT_SOURCE_DIR}/ncnnlib/lib/libncnn.a
        )


############### source and test ######################
# 添加包含文件的目录路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ncnnlib/include)
# CMake 将进入这些子目录，并执行各自的 CMakeLists.txt 文件以构建源代码和测试代码
add_subdirectory(src)
add_subdirectory(app)
