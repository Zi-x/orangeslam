#pragma once
#ifndef ORANGESLAM_CONFIG_H
#define ORANGESLAM_CONFIG_H

#include "orangeslam/common_include.h"

namespace orangeslam {

/**
 * 配置类，使用SetParameterFile确定配置文件
 * 然后用Get得到对应值
 * 单例模式
 */
class Config {
   private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {}  // private constructor makes a singleton
   public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T Get(const std::string &key) {
        T data_;  
        // LOG(INFO) << "Init2.  " << key;
        // std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
        Config::config_->file_[key] >> data_;
        // LOG(INFO) << "Init2.1";
        return data_;
    }

    // template <typename T>
    // static T Get(const std::string &key) {
    //     return T(Config::config_->file_[key]);
    // }
};
}  // namespace orangeslam

#endif  // ORANGESLAM_CONFIG_H
