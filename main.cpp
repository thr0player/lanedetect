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
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>


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

        void LidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_cloud_msg,
                           const geometry_msgs::PoseStampedConstPtr &imu_msg);
    };

    DetectManager::DetectManager() : node_handle_("~") {
        ROS_INFO("Inititalizing BucketFiltering node ...");
        node_handle_.param<std::string>("point_topic", point_topic_, "/velodyne_points");
        ROS_INFO("Input Point Cloud: %s", point_topic_.c_str());

        node_handle_.param<std::string>("imu_topic", imu_topic_, "/pose_imu");
        ROS_INFO("Input imu Cloud: %s", imu_topic_.c_str());

//        filtered_points_pub_ = node_handle_.advertise < pcl::PointCloud < pcl::PointXYZI >> (filtered_points_topic_, 10000);

        message_filters::Subscriber <sensor_msgs::PointCloud2> lidar_sub(node_handle_, point_topic_, 1000);
        message_filters::Subscriber <geometry_msgs::PoseStamped> imu_sub(node_handle_, imu_topic_, 1000);
        message_filters::Synchronizer <sync_policy_classification> sync(sync_policy_classification(100), lidar_sub,
                                                                        imu_sub);

        sync.registerCallback(boost::bind(&DetectManager::LidarCallback, this, _1, _2));

        ros::spin();

    }

    void DetectManager::LidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_cloud_msg,
                                      const geometry_msgs::PoseStampedConstPtr &imu_msg) {

        ROS_INFO("I heard the pose from the robot");
        ROS_INFO("the position(x,y,z) is %f , %f, %f", imu_msg->pose.position.x, imu_msg->pose.position.y, imu_msg->pose.position.z);
        ROS_INFO("the orientation(x,y,z,w) is %f , %f, %f, %f", imu_msg->pose.orientation.x, imu_msg->pose.orientation.y, imu_msg->pose.orientation.z, imu_msg->pose.orientation.w);
        ROS_INFO("the time we get the pose is %f",  imu_msg->header.stamp.sec + 1e-9*imu_msg->header.stamp.nsec);
        ROS_INFO("the lidar size is %d",  in_cloud_msg->height*in_cloud_msg->width);

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
