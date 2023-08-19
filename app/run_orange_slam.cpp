#include <gflags/gflags.h>
#include "orangeslam/visual_odometry.h"

DEFINE_string(config_file, "../config/default.yaml", "config file");

int main(int argc, char **argv) {
    
    google::ParseCommandLineFlags(&argc, &argv, true);

    orangeslam::VisualOdometry::Ptr vo(
        new orangeslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);

    
    vo->Run();
    
    return 0;
}
