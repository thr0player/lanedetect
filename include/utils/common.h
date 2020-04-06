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

#define IMAGE_HEIGHT 600
#define IMAGE_WIDTH 2000
#define RES_STEP 0.04

    typedef pcl::PointXYZI PPoint;

    class Frame{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pcl::PointCloud<PPoint>::Ptr lidar;
        Eigen::Matrix4f pose;

    };

}
#endif //LANEDETECTION_COMMON_H
