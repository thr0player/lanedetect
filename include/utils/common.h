//
// Created by lw on 20-4-6.
//

#ifndef LANEDETECTION_COMMON_H
#define LANEDETECTION_COMMON_H

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <assert.h>

namespace LaneDetect {

    const int IMAGE_HEIGHT = 600;
    const int IMAGE_WIDTH = 2000;
    const float RES_STEP = 0.05;

    const float EPS = 1e-6;

    typedef pcl::PointXYZI PPoint;

    class Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pcl::PointCloud<PPoint>::Ptr lidar;
        Eigen::Matrix4f pose;

    };

}
#endif //LANEDETECTION_COMMON_H
