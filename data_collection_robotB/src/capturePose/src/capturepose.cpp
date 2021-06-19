// Author:  Junyi Ma     mjy980625@163.com


#include <math.h>
#include <vector>
#include <capturePose/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <unordered_map>

#include "capturePose/common.h"
#include "capturePose/tic_toc.h"
#include  <pcl_ros/point_cloud.h>
#include <iostream>
#include <fstream>
#include <string>

int frameCount = 0;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;
std::mutex mPFH;
std::string line;
std::string line2;

std::size_t line_num = 0;
std::vector<float> gtTimeVect;
std::vector<Eigen::Matrix<double, 3, 4>> gtPoseVec;
nav_msgs::Path laserAfterMappedPath;
std::ofstream odomAftRecord("/home/mjy/dev/aloam/pose/truePose.txt");
std::ofstream timeAftRecord("/home/mjy/dev/aloam/pose/timePose.txt");

#include<mutex>


int count_save=0;

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	count_save++;
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;


	double thisPoseTime = laserOdometry->header.stamp.toSec();


	Eigen::Vector3d eularAngle = q_wodom_curr.matrix().eulerAngles(0,1,2);

	odomAftRecord <<count_save <<" "<< t_wodom_curr.x() <<" "<< t_wodom_curr.y()<<" " << t_wodom_curr.z()<<" "<<laserOdometry->pose.pose.orientation.x<<" "<<laserOdometry->pose.pose.orientation.y<<" "<<laserOdometry->pose.pose.orientation.z<<" "<<laserOdometry->pose.pose.orientation.w <<" "<< eularAngle.x()<<" " << eularAngle.y() <<" "<< eularAngle.z() << "\n";

	timeAftRecord  << count_save<<" " << laserOdometry->header.stamp << "\n";
	mBuf.unlock();
}






int main(int argc, char **argv)
{
	ros::init(argc, argv, "capturepose");
	ros::NodeHandle nh;

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/robot2/aft_mapped_to_init", 100, laserOdometryHandler);

	ros::spin();

	return 0;
}
