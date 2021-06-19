cmake_minimum_required(VERSION 2.8.3)
project(capturePose)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  pcl_conversions
)

# include_directories("/usr/include/eigen3")
# include_directories("/usr/local/include/eigen3")
# set(PCL_DIR "/usr/local/pcl-1.8/share/pcl-1.8")
# set(PCL_DIR "/usr/share/pcl-1.10")
include_directories("/usr/local/include/eigen3")
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs pcl_conversions
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)




add_executable(capturepose src/capturepose.cpp)
target_link_libraries(capturepose ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} )



