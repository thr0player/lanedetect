//
// Created by lw on 20-4-6.
//

#ifndef LANEDETECTION_UTILS_HPP
#define LANEDETECTION_UTILS_HPP

#include <Eigen/Geometry>
#include <Eigen/Core>

namespace LaneDetect {
    Eigen::Matrix4f ToMatrix(const Eigen::Matrix3f& rot, const Eigen::Vector3f& vec){
        Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
        mat.block<3,3>(0,0) = rot;
        mat.block<3,1>(0,3) = vec;
        return mat;
    }


}
#endif //LANEDETECTION_UTILS_HPP
