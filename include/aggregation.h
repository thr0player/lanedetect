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
        Aggregation() {
            img = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
            avgimg = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);
            countimg = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);
            imgzmin = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);
            imgzmax= cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);
        };

        void Process(const std::deque<Frame> &data_buffer, const Eigen::Matrix4f &world_imu, const Eigen::Matrix4f &imu_vel,
                     int frame_size, pcl::PointCloud<PPoint>::Ptr &pc, pcl::PointCloud<PPoint>::Ptr& labelpc);

        void ToPclPc(pcl::PointCloud<PPoint>::Ptr &pc, pcl::PointCloud<pcl::PointXYZ>::Ptr &pclpc);

    private:
        cv::Mat img; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
        cv::Mat avgimg; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);
        cv::Mat countimg; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);
        cv::Mat imgzmin; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);
        cv::Mat imgzmax; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32SC1);

    };

}
#endif //LANEDETECTION_AGGREGATION_H
