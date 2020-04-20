//
// Created by lw on 20-4-6.
//

#include "aggregation.h"
#include <pcl/common/transforms.h>                  //allows us to use pcl::transformPointCloud function

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "cca.cc"
#include <map>

#include <assert.h>
namespace LaneDetect {

void Aggregation::Process(const std::deque<Frame> &data_buffer, const Eigen::Matrix4f &world_imu, const Eigen::Matrix4f &imu_vel,
             int frame_size, pcl::PointCloud<PPoint>::Ptr& pc, pcl::PointCloud<PPoint>::Ptr& labelpc) {

    if (data_buffer.size() <= frame_size) {
        return;
    }
    const float cx = IMAGE_WIDTH * 0.5;
    const float cy = IMAGE_HEIGHT * 0.5;

    pc->clear();
    Eigen::Matrix4f startpose = data_buffer.at(data_buffer.size() - 1).pose;
    for (size_t di = data_buffer.size() - 1; di >= 0 && di >= data_buffer.size() - frame_size; di--) {
        pcl::PointCloud<PPoint> localpc;
        Eigen::Matrix4f curpose = startpose.inverse() * data_buffer.at(di).pose * imu_vel;
        pcl::transformPointCloud(*data_buffer.at(di).lidar, localpc, curpose);

//        ROS_INFO("[Aggregation]: cur %zu size %zu ", di, pc->size());
        (*pc) += (localpc);
    }

    ROS_INFO("[Aggregation]: total size %zu ", pc->size());

//    cv<int> count(IMAGE_HEIGHT * IMAGE_WIDTH, 0);

    img.setTo(0);
    avgimg.setTo(0);
    countimg.setTo(0);

    imgzmax.setTo(-99999);
    imgzmin.setTo(999999);

    auto AvgImg = [&pc, this,cx,cy]() {
        for (auto &pi:pc->points) {
            int col = round(pi.x / RES_STEP + cx);
            int row = round(-pi.y / RES_STEP + cy);

            if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT - 1) {
                continue;
            }

            if (pi.z > imgzmax.at<float>(row, col)) {
                imgzmax.at<float>(row, col) = pi.z;
            }

            if (pi.z < imgzmin.at<float>(row, col)) {
                imgzmin.at<float>(row, col) = pi.z;
            }

//            if (pi.intensity * 255 < 100) continue;

            avgimg.at<float>((row), (col)) += pi.intensity;
            countimg.at<int>((row), (col))++;
        }

        for (int row = 0; row < IMAGE_HEIGHT; ++row) {
            for (int col = 0; col < IMAGE_WIDTH; ++col) {
                if (countimg.at<int>(row, col) < 5) continue;
                avgimg.at<float>(row, col) /= countimg.at<int>(row, col);

                if (avgimg.at<float>(row, col) * 255 > 50)
//                    if (imgzmax.at<float>(row, col) - imgzmin.at<float>(row, col) < 0.4)
                    img.at<uchar>(row, col) = avgimg.at<float>(row, col) * 255;

            }
        }
    };

    auto ProjImg = [&pc, this,cx,cy]() {

        for (auto &pi:pc->points) {
            float col = pi.x / RES_STEP + cx;
            float row = -pi.y / RES_STEP + cy;

            if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT - 1) {
                continue;
            }

            if (pi.intensity * 255 < 1) continue;
            img.at<uchar>(round(row), round(col)) = pi.intensity * 255;

        }

    };

    auto MaxImg = [&pc, this,cx,cy]() {

        for (auto &pi:pc->points) {
            int col = round(pi.x / RES_STEP + cx);
            int row = round(-pi.y / RES_STEP + cy);

            if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT - 1) {
                continue;
            }

            if (pi.z > imgzmax.at<float>(row, col)) {
                imgzmax.at<float>(row, col) = pi.z;
            }

            if (pi.z < imgzmin.at<float>(row, col)) {
                imgzmin.at<float>(row, col) = pi.z;
            }

            if (pi.intensity * 255 < 100) continue;

            img.at<uchar>((row), (col)) = pi.intensity * 255;
        }


        for (int row = 0; row < IMAGE_HEIGHT; ++row) {
            for (int col = 0; col < IMAGE_WIDTH; ++col) {

                if (imgzmax.at<float>(row, col) - imgzmin.at<float>(row, col) > 0.2)
                    img.at<uchar>(row, col) = 0;

            }
        }

    };

//    ProjImg();
//    AvgImg();
    MaxImg();

    ////filter image
    bool enableshowimage = false;
    if (enableshowimage ) {
        cv::imshow("img", img);
    }
    cv::medianBlur(img, img, 3);
    cv::dilate(img, img, 3);

    if (enableshowimage ) {
        cv::imshow("filter img", img);
//        cv::waitKey(60);
    }

    //////part.2 filter pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr fpc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<PPoint>::Ptr new_fpc(new pcl::PointCloud<PPoint>());

    for (auto &pi:pc->points) {
        int col = round(pi.x / RES_STEP + cx);
        int row = round(-pi.y / RES_STEP + cy);

        if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT - 1) {
            continue;
        }

        if (img.at<uchar>(row, col) > 0) {
            new_fpc->points.emplace_back(pi);
        }
    }
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_bak(new pcl::PointCloud<pcl::PointXYZ>());
//    ToPclPc(pc, pc_bak);
//
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//    tree->setInputCloud(pc_bak);
//
//    const float r = 0.2;
//    for (auto &pp: fpc->points) {
//        std::vector<int> pointIdxNKNSearch;
//        std::vector<float> pointNKNSquaredDistance;
//
//        tree->radiusSearch(pp, r, pointIdxNKNSearch, pointNKNSquaredDistance);
//        if (pointIdxNKNSearch.size() > 0) {
//
//            for (auto &pidx:pointIdxNKNSearch) {
//                if (pc->points.at(pidx).intensity * 255 > 30)
//                    new_fpc->points.emplace_back(pc->points.at(pidx));
//            }
//
//        }else{
//            std::cout<<"[WARNING] not found"<<std::endl;
//        }
//    }

    auto ProjToImg=[this,cx,cy](cv::Mat&img, pcl::PointCloud<PPoint>::Ptr& pc){
        img.setTo(0);
        for (auto &pi:pc->points) {

            int col = round(pi.x / RES_STEP + cx);
            int row = round(-pi.y / RES_STEP + cy);

            if (col < 0 || col > IMAGE_WIDTH - 1 || row < 0 || row > IMAGE_HEIGHT - 1) {
                continue;
            }

            if (img.at<uchar>(row, col) < pi.intensity*255) {
                img.at<uchar>(row, col) = pi.intensity*255;
            }
        }
    };

    if (enableshowimage ) {
        cv::Mat showimg = img.clone();
        ProjToImg(showimg, new_fpc);
        cv::imshow("final img", showimg);
        cv::waitKey(60);
    }

    pc = new_fpc;

    ///// clustering
    cv::Mat _lableImg;
    std::set<int> labels;


    ros::Time t1=ros::Time::now();
    CcaByTwoPass(img, _lableImg, labels);
    ros::Duration elap_timet = ros::Time::now()-t1;
    double dur = elap_timet.toSec();
    ROS_INFO("CCA time: %f",dur);

    std::map<int,int> label_map;
    for(auto &si:labels){
        label_map[si] = label_map.size();
    }

    ROS_INFO("clustered label size: %zu",labels.size());

    std::vector<cv::Vec3b> color_table;
    for (size_t ci = 0; ci < labels.size()+10; ++ci) {
        color_table.emplace_back(cv::Vec3b(rand()%255, rand()%255, rand()%255));
    }
    cv::Mat color_label;
    cv::cvtColor(img, color_label, CV_GRAY2BGR);


    for (int row = 0; row < _lableImg.rows; ++row) {
        for (int col = 0; col < _lableImg.cols; ++col) {
            const int &label = _lableImg.at<int>(row, col);
            if(label !=0 ){
                int idx = label_map[label];
                color_label.at<cv::Vec3b>(row,col)[0] = color_table.at(idx).val[0];
                color_label.at<cv::Vec3b>(row,col)[1] = color_table.at(idx).val[1];
                color_label.at<cv::Vec3b>(row,col)[2] = color_table.at(idx).val[2];
            }
        }
    }
//    cv::imshow("_lableImg",_lableImg);

    cv::imshow("color_label",color_label);
    cv::waitKey(30);
}

void Aggregation::ToPclPc(pcl::PointCloud<PPoint>::Ptr &pc, pcl::PointCloud<pcl::PointXYZ>::Ptr &pclpc) {
    assert(pc != nullptr);
    assert(pclpc != nullptr);
    pclpc->clear();

    for (auto &pi:pc->points) {
        pclpc->points.emplace_back(pcl::PointXYZ(pi.x, pi.y, pi.z));
    }

}


}//namespace