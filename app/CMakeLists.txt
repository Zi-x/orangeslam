add_executable(run_orange_slam run_orange_slam.cpp)
target_link_libraries(run_orange_slam orangeslam ${THIRD_PARTY_LIBS})