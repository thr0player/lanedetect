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
    const float RES_STEP = 0.08;

    const float EPS = 1e-6;

    typedef pcl::PointXYZI PPoint;

    class Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pcl::PointCloud<PPoint>::Ptr lidar;
        Eigen::Matrix4f pose;

    };

#define  REPEAT_TEST_EXIT(...) \
  do \
  { \
    static int hit = 0; \
    if (hit > (__VA_ARGS__)) \
    { \
      exit(0); \
    } \
    hit++; \
    std::cout<<"r_test_"<<hit<<std::endl; \
  } while(0)


}
#endif //LANEDETECTION_COMMON_H
