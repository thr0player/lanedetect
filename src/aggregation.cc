//
// Created by lw on 20-4-6.
//

#include "aggregation.h"
#include <pcl/common/transforms.h>                  //allows us to use pcl::transformPointCloud function
#include <assert.h>
namespace LaneDetect {

void Aggregation::Process(const std::deque<Frame> &data_buffer, const Eigen::Matrix4f &world_imu, const Eigen::Matrix4f &imu_vel,
             int frame_size, pcl::PointCloud<PPoint>::Ptr& pc){

    if(data_buffer.size() <= frame_size){
        return;
    }

    pc->clear();
    Eigen::Matrix4f startpose = data_buffer.at(data_buffer.size() - 1).pose;
    for (size_t di = data_buffer.size() - 1; di >= 0 && di >= data_buffer.size() - frame_size; di--) {
        pcl::PointCloud<PPoint> localpc;
        Eigen::Matrix4f curpose = startpose.inverse() * data_buffer.at(di).pose * imu_vel;
        pcl::transformPointCloud(*data_buffer.at(di).lidar, localpc, curpose);

        ROS_INFO("[Aggregation]: cur %zu size %zu ", di,  pc->size());

        (*pc) += (localpc);
    }

    ROS_INFO("[Aggregation]: total size %zu ", pc->size());


    cv::Mat img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
    img.setTo(0);

    auto ProjImg = [&pc,&img]() {

        const float cx = IMAGE_WIDTH * 0.5;
        const float cy = IMAGE_HEIGHT * 0.5;
        for (auto &pi:pc->points) {
            float col = pi.x / RES_STEP + cx;
            float row = -pi.y / RES_STEP + cy;

            if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT-1) {
                continue;
            }

            img.at<uchar>(round(row), round(col)) = pi.intensity*255;

        }

    };

    ProjImg();

    cv::imshow("img", img);
    cv::waitKey(60);


}

}