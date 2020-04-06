//
// Created by lw on 20-3-29.
//
#include <iostream>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <deque>

#include "utils/utils.hpp"
#include "utils/common.h"

#include "aggregation.h"

namespace LaneDetect {

    class DetectManager {
    public:
        typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::PointCloud2, geometry_msgs::PoseStamped> sync_policy_classification;

        DetectManager();

    private:

        ros::NodeHandle node_handle_;
        ros::Subscriber points_node_sub_;
        ros::Publisher filtered_points_pub_;

        std::string point_topic_;
        std::string imu_topic_;
        std::string filtered_points_topic_;

        ros::Time t1_;
        ros::Time t2_;
        ros::Duration elap_time_;

        Aggregation aggregation_;
        int lidar_size_;
        std::deque<Frame> data_buffer_;

        Eigen::Matrix4f world_imu_;
        Eigen::Matrix4f imu_vel_;

        void ListenTf();

        void LidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_cloud_msg,
                           const geometry_msgs::PoseStampedConstPtr &imu_msg);
    };

    DetectManager::DetectManager() : node_handle_("~"), lidar_size_(30) {
        ROS_INFO("Inititalizing BucketFiltering node ...");
        node_handle_.param<std::string>("point_topic", point_topic_, "/velodyne_points");
        ROS_INFO("Input Point Cloud: %s", point_topic_.c_str());

        node_handle_.param<std::string>("imu_topic", imu_topic_, "/pose_imu");
        ROS_INFO("Input imu Cloud: %s", imu_topic_.c_str());

        node_handle_.param<std::string>("outlidar_topic", filtered_points_topic_, "/agg_lidar");
        ROS_INFO("output lidar Cloud: %s", filtered_points_topic_.c_str());

        filtered_points_pub_ = node_handle_.advertise<pcl::PointCloud<PPoint>>(filtered_points_topic_, 10000);

        message_filters::Subscriber <sensor_msgs::PointCloud2> lidar_sub(node_handle_, point_topic_, 1000);
        message_filters::Subscriber <geometry_msgs::PoseStamped> imu_sub(node_handle_, imu_topic_, 1000);
        message_filters::Synchronizer <sync_policy_classification> sync(sync_policy_classification(100), lidar_sub,
                                                                        imu_sub);

        sync.registerCallback(boost::bind(&DetectManager::LidarCallback, this, _1, _2));

        ListenTf();

        ros::spin();

    }

    void DetectManager::ListenTf(){
        tf::TransformListener listener;
        ROS_INFO("wait for tf: world imu");
        ros::Rate rate(10.0);
        while (node_handle_.ok()){
            tf::StampedTransform transform;
            try{
                listener.waitForTransform("/world","/imu",ros::Time(0),ros::Duration(3.0));
                listener.lookupTransform("/world", "/imu",
                                         ros::Time(0), transform);
            }
            catch (tf::TransformException &ex) {
                ROS_ERROR("%s",ex.what());
                ros::Duration(1.0).sleep();
                continue;
            }

            Eigen::Quaternionf rot(transform.getRotation().w(),transform.getRotation().x(),transform.getRotation().y(),transform.getRotation().z());
            Eigen::Vector3f vec; vec << transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z();
            world_imu_ = ToMatrix(rot.toRotationMatrix(), vec);
            std::cout << "tf get :world - imu ：\n" << world_imu_ << std::endl;
            break;
            rate.sleep();
        }

        ROS_INFO("wait for tf: world vel");
        while (node_handle_.ok()){
            tf::StampedTransform transform;
            try{
                listener.waitForTransform("/imu","/velodyne",ros::Time(0),ros::Duration(3.0));
                listener.lookupTransform("/imu", "/velodyne",
                                         ros::Time(0), transform);
            }
            catch (tf::TransformException &ex) {
                ROS_ERROR("%s",ex.what());
                ros::Duration(1.0).sleep();
                continue;
            }

            Eigen::Quaternionf rot(transform.getRotation().w(),transform.getRotation().x(),transform.getRotation().y(),transform.getRotation().z());
            Eigen::Vector3f vec; vec << transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z();
            imu_vel_ = ToMatrix(rot.toRotationMatrix(), vec);

            std::cout << "tf get :imu - vel：\n" << imu_vel_<< std::endl;
            break;
            rate.sleep();
        }


    }


    void DetectManager::LidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_cloud_msg,
                                      const geometry_msgs::PoseStampedConstPtr &imu_msg) {

//        ROS_INFO("I heard the pose from the robot");
//        ROS_INFO("the position(x,y,z) is %f , %f, %f", imu_msg->pose.position.x, imu_msg->pose.position.y, imu_msg->pose.position.z);
//        ROS_INFO("the orientation(x,y,z,w) is %f , %f, %f, %f", imu_msg->pose.orientation.x, imu_msg->pose.orientation.y, imu_msg->pose.orientation.z, imu_msg->pose.orientation.w);
//        ROS_INFO("the time we get the pose is %f",  imu_msg->header.stamp.sec + 1e-9*imu_msg->header.stamp.nsec);
//        ROS_INFO("the lidar size is %d",  in_cloud_msg->height*in_cloud_msg->width);
        t1_ = ros::Time().now();

        pcl::PointCloud<PPoint>::Ptr c_in_cloud(new pcl::PointCloud<PPoint>());
        pcl::fromROSMsg(*in_cloud_msg, *c_in_cloud);

        Eigen::Vector3f vec; vec << imu_msg->pose.position.x, imu_msg->pose.position.y, imu_msg->pose.position.z;
        Eigen::Quaternionf rot(imu_msg->pose.orientation.w, imu_msg->pose.orientation.x, imu_msg->pose.orientation.y,
                               imu_msg->pose.orientation.z);

        Frame local;
        local.lidar = c_in_cloud;
        local.pose = ToMatrix(rot.toRotationMatrix(), vec);

        if (data_buffer_.size() > static_cast<size_t>(lidar_size_)) {
            data_buffer_.pop_front();
        }

        data_buffer_.emplace_back(local);


        pcl::PointCloud<PPoint>::Ptr out_pc(new pcl::PointCloud<PPoint>());
        aggregation_.Process(data_buffer_, world_imu_, imu_vel_, 20, out_pc);

        out_pc->header = c_in_cloud->header;
        filtered_points_pub_.publish(out_pc);
        t2_ = ros::Time().now();
        elap_time_ = t2_ - t1_;

        ROS_INFO("finish aggregation : size %zu,  time cluster %f s", out_pc->size(), elap_time_.toSec());

    }

}

int main(int argc, char **argv)
{
    std::cout<<"Inititalizing now !!!!"<<std::endl;
    ros::init(argc, argv, "LaneDetection");
    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info) ) {
        ros::console::notifyLoggerLevelsChanged();
    }

    LaneDetect::DetectManager manager_;

    while(ros::ok()){
        ros::spinOnce();
    }
    ROS_INFO("exit now");
    return 0;

}
