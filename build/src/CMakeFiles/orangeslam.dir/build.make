# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/orangepi/orangeslam/orangeslam_modify2_backup

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/orangepi/orangeslam/orangeslam_modify2_backup/build

# Include any dependencies generated for this target.
include src/CMakeFiles/orangeslam.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/orangeslam.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/orangeslam.dir/flags.make

src/CMakeFiles/orangeslam.dir/frame.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/frame.cpp.o: ../src/frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/orangeslam.dir/frame.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/frame.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frame.cpp

src/CMakeFiles/orangeslam.dir/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/frame.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frame.cpp > CMakeFiles/orangeslam.dir/frame.cpp.i

src/CMakeFiles/orangeslam.dir/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/frame.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frame.cpp -o CMakeFiles/orangeslam.dir/frame.cpp.s

src/CMakeFiles/orangeslam.dir/mappoint.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/mappoint.cpp.o: ../src/mappoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/orangeslam.dir/mappoint.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/mappoint.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/mappoint.cpp

src/CMakeFiles/orangeslam.dir/mappoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/mappoint.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/mappoint.cpp > CMakeFiles/orangeslam.dir/mappoint.cpp.i

src/CMakeFiles/orangeslam.dir/mappoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/mappoint.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/mappoint.cpp -o CMakeFiles/orangeslam.dir/mappoint.cpp.s

src/CMakeFiles/orangeslam.dir/map.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/map.cpp.o: ../src/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/orangeslam.dir/map.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/map.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/map.cpp

src/CMakeFiles/orangeslam.dir/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/map.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/map.cpp > CMakeFiles/orangeslam.dir/map.cpp.i

src/CMakeFiles/orangeslam.dir/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/map.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/map.cpp -o CMakeFiles/orangeslam.dir/map.cpp.s

src/CMakeFiles/orangeslam.dir/camera.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/orangeslam.dir/camera.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/camera.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/camera.cpp

src/CMakeFiles/orangeslam.dir/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/camera.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/camera.cpp > CMakeFiles/orangeslam.dir/camera.cpp.i

src/CMakeFiles/orangeslam.dir/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/camera.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/camera.cpp -o CMakeFiles/orangeslam.dir/camera.cpp.s

src/CMakeFiles/orangeslam.dir/config.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/orangeslam.dir/config.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/config.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/config.cpp

src/CMakeFiles/orangeslam.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/config.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/config.cpp > CMakeFiles/orangeslam.dir/config.cpp.i

src/CMakeFiles/orangeslam.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/config.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/config.cpp -o CMakeFiles/orangeslam.dir/config.cpp.s

src/CMakeFiles/orangeslam.dir/feature.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/feature.cpp.o: ../src/feature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/orangeslam.dir/feature.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/feature.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/feature.cpp

src/CMakeFiles/orangeslam.dir/feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/feature.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/feature.cpp > CMakeFiles/orangeslam.dir/feature.cpp.i

src/CMakeFiles/orangeslam.dir/feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/feature.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/feature.cpp -o CMakeFiles/orangeslam.dir/feature.cpp.s

src/CMakeFiles/orangeslam.dir/frontend.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/frontend.cpp.o: ../src/frontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/orangeslam.dir/frontend.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/frontend.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frontend.cpp

src/CMakeFiles/orangeslam.dir/frontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/frontend.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frontend.cpp > CMakeFiles/orangeslam.dir/frontend.cpp.i

src/CMakeFiles/orangeslam.dir/frontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/frontend.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/frontend.cpp -o CMakeFiles/orangeslam.dir/frontend.cpp.s

src/CMakeFiles/orangeslam.dir/backend.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/backend.cpp.o: ../src/backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/orangeslam.dir/backend.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/backend.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/backend.cpp

src/CMakeFiles/orangeslam.dir/backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/backend.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/backend.cpp > CMakeFiles/orangeslam.dir/backend.cpp.i

src/CMakeFiles/orangeslam.dir/backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/backend.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/backend.cpp -o CMakeFiles/orangeslam.dir/backend.cpp.s

src/CMakeFiles/orangeslam.dir/viewer.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/viewer.cpp.o: ../src/viewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/orangeslam.dir/viewer.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/viewer.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/viewer.cpp

src/CMakeFiles/orangeslam.dir/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/viewer.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/viewer.cpp > CMakeFiles/orangeslam.dir/viewer.cpp.i

src/CMakeFiles/orangeslam.dir/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/viewer.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/viewer.cpp -o CMakeFiles/orangeslam.dir/viewer.cpp.s

src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.o: ../src/visual_odometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/visual_odometry.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/visual_odometry.cpp

src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/visual_odometry.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/visual_odometry.cpp > CMakeFiles/orangeslam.dir/visual_odometry.cpp.i

src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/visual_odometry.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/visual_odometry.cpp -o CMakeFiles/orangeslam.dir/visual_odometry.cpp.s

src/CMakeFiles/orangeslam.dir/dataset.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/orangeslam.dir/dataset.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/dataset.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/dataset.cpp

src/CMakeFiles/orangeslam.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/dataset.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/dataset.cpp > CMakeFiles/orangeslam.dir/dataset.cpp.i

src/CMakeFiles/orangeslam.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/dataset.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/dataset.cpp -o CMakeFiles/orangeslam.dir/dataset.cpp.s

src/CMakeFiles/orangeslam.dir/fdetect.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/fdetect.cpp.o: ../src/fdetect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/CMakeFiles/orangeslam.dir/fdetect.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/fdetect.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/fdetect.cpp

src/CMakeFiles/orangeslam.dir/fdetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/fdetect.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/fdetect.cpp > CMakeFiles/orangeslam.dir/fdetect.cpp.i

src/CMakeFiles/orangeslam.dir/fdetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/fdetect.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/fdetect.cpp -o CMakeFiles/orangeslam.dir/fdetect.cpp.s

src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o: src/CMakeFiles/orangeslam.dir/flags.make
src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o: ../src/yolofastestv2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o -c /home/orangepi/orangeslam/orangeslam_modify2_backup/src/yolofastestv2.cpp

src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orangeslam.dir/yolofastestv2.cpp.i"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/orangepi/orangeslam/orangeslam_modify2_backup/src/yolofastestv2.cpp > CMakeFiles/orangeslam.dir/yolofastestv2.cpp.i

src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orangeslam.dir/yolofastestv2.cpp.s"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/orangepi/orangeslam/orangeslam_modify2_backup/src/yolofastestv2.cpp -o CMakeFiles/orangeslam.dir/yolofastestv2.cpp.s

# Object files for target orangeslam
orangeslam_OBJECTS = \
"CMakeFiles/orangeslam.dir/frame.cpp.o" \
"CMakeFiles/orangeslam.dir/mappoint.cpp.o" \
"CMakeFiles/orangeslam.dir/map.cpp.o" \
"CMakeFiles/orangeslam.dir/camera.cpp.o" \
"CMakeFiles/orangeslam.dir/config.cpp.o" \
"CMakeFiles/orangeslam.dir/feature.cpp.o" \
"CMakeFiles/orangeslam.dir/frontend.cpp.o" \
"CMakeFiles/orangeslam.dir/backend.cpp.o" \
"CMakeFiles/orangeslam.dir/viewer.cpp.o" \
"CMakeFiles/orangeslam.dir/visual_odometry.cpp.o" \
"CMakeFiles/orangeslam.dir/dataset.cpp.o" \
"CMakeFiles/orangeslam.dir/fdetect.cpp.o" \
"CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o"

# External object files for target orangeslam
orangeslam_EXTERNAL_OBJECTS =

../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/frame.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/mappoint.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/map.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/camera.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/config.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/feature.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/frontend.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/backend.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/viewer.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/visual_odometry.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/dataset.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/fdetect.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/yolofastestv2.cpp.o
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/build.make
../lib/liborangeslam.so: /usr/local/lib/libopencv_dnn.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_highgui.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_ml.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_objdetect.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_shape.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_stitching.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_superres.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_videostab.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_viz.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libpango_glgeometry.so
../lib/liborangeslam.so: /usr/local/lib/libpango_plot.so
../lib/liborangeslam.so: /usr/local/lib/libpango_python.so
../lib/liborangeslam.so: /usr/local/lib/libpango_scene.so
../lib/liborangeslam.so: /usr/local/lib/libpango_tools.so
../lib/liborangeslam.so: /usr/local/lib/libpango_video.so
../lib/liborangeslam.so: /usr/local/lib/libglog.so
../lib/liborangeslam.so: /usr/local/lib/libgflags.a
../lib/liborangeslam.so: /usr/lib/aarch64-linux-gnu/libcxsparse.so
../lib/liborangeslam.so: ../ncnnlib/lib/libncnn.a
../lib/liborangeslam.so: /usr/local/lib/libopencv_calib3d.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_features2d.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_flann.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_photo.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_video.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_videoio.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_imgproc.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libopencv_core.so.3.4.16
../lib/liborangeslam.so: /usr/local/lib/libpango_geometry.so
../lib/liborangeslam.so: /usr/local/lib/libtinyobj.so
../lib/liborangeslam.so: /usr/local/lib/libpango_display.so
../lib/liborangeslam.so: /usr/local/lib/libpango_vars.so
../lib/liborangeslam.so: /usr/local/lib/libpango_windowing.so
../lib/liborangeslam.so: /usr/local/lib/libpango_opengl.so
../lib/liborangeslam.so: /usr/lib/aarch64-linux-gnu/libGLEW.so
../lib/liborangeslam.so: /usr/lib/aarch64-linux-gnu/libOpenGL.so
../lib/liborangeslam.so: /usr/lib/aarch64-linux-gnu/libGLX.so
../lib/liborangeslam.so: /usr/lib/aarch64-linux-gnu/libGLU.so
../lib/liborangeslam.so: /usr/local/lib/libpango_image.so
../lib/liborangeslam.so: /usr/local/lib/libpango_packetstream.so
../lib/liborangeslam.so: /usr/local/lib/libpango_core.so
../lib/liborangeslam.so: /usr/local/lib/libfmt.so.10.0.1
../lib/liborangeslam.so: src/CMakeFiles/orangeslam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/orangepi/orangeslam/orangeslam_modify2_backup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX shared library ../../lib/liborangeslam.so"
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orangeslam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/orangeslam.dir/build: ../lib/liborangeslam.so

.PHONY : src/CMakeFiles/orangeslam.dir/build

src/CMakeFiles/orangeslam.dir/clean:
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src && $(CMAKE_COMMAND) -P CMakeFiles/orangeslam.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/orangeslam.dir/clean

src/CMakeFiles/orangeslam.dir/depend:
	cd /home/orangepi/orangeslam/orangeslam_modify2_backup/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/orangepi/orangeslam/orangeslam_modify2_backup /home/orangepi/orangeslam/orangeslam_modify2_backup/src /home/orangepi/orangeslam/orangeslam_modify2_backup/build /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src /home/orangepi/orangeslam/orangeslam_modify2_backup/build/src/CMakeFiles/orangeslam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/orangeslam.dir/depend
