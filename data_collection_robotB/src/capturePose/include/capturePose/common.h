#pragma once

#include <cmath>

#include <pcl/point_types.h>

#include<chrono>
#include<ros/ros.h>
#include<ros/time.h>
#include <iostream>
#include<fstream>

using Time = std::chrono::system_clock::time_point;

typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}


// ROS time adapters
// 从1970年1月1日开始算，所以是epoch + since_epoch
inline Time fromROSTime(ros::Time const& rosTime)
{
  auto epoch = std::chrono::system_clock::time_point();
  auto since_epoch = std::chrono::seconds(rosTime.sec) + std::chrono::nanoseconds(rosTime.nsec);
  return epoch + since_epoch;
}

inline ros::Time toROSTime(Time const& time_point)
{
  return ros::Time().fromNSec(std::chrono::duration_cast<std::chrono::nanoseconds>(time_point.time_since_epoch()).count());
}




