//
// Created by lw on 20-4-6.
//

#ifndef LANEDETECTION_AGGREGATION_H
#define LANEDETECTION_AGGREGATION_H

#include "utils/common.h"
#include <deque>
#include <opencv2/opencv.hpp>

namespace LaneDetect {

    class Aggregation {
    public:
        Aggregation() {};

        void Process(const std::deque<Frame> &data_buffer, const Eigen::Matrix4f &world_imu, const Eigen::Matrix4f &imu_vel,
                     int frame_size, pcl::PointCloud<PPoint>::Ptr &pc);

    };

}
#endif //LANEDETECTION_AGGREGATION_H
