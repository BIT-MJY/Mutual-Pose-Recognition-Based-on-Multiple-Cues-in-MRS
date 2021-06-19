#include <cmath>
#include <vector>
#include <string>
#include "cal_vfh/common.h"
#include "cal_vfh/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
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

 
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <sstream>

#include<mutex>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

int N_SCANS = 0;
double MINIMUM_RANGE = 0.1; 
double VEHICLE_RANGE_X_MIN = -4;
double VEHICLE_RANGE_X_MAX = 4;
double VEHICLE_RANGE_Y_MIN = -1;
double VEHICLE_RANGE_Y_MAX = 6;
double VEHICLE_RANGE_Z_MIN = -1.2;
double VEHICLE_RANGE_Z_MAX = 0.1;
double d_th = 5;
std::string PATH_PT = "/media/mjy/Samsung_T5/linux/ICUS2021/data/vfh/";
std::string PATH_VFH = "/media/mjy/Samsung_T5/linux/ICUS2021/data/vfh_txt/VFH.txt";
std::string PATH_TIME = "/media/mjy/Samsung_T5/linux/ICUS2021/data/vfh_txt/time.txt";

int count_save = 0;

ros::Publisher pubLaserVehicle;

std::ofstream vfhRecord(PATH_VFH);
std::ofstream timeRecord(PATH_TIME);



void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{

    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudVehicle(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i=0; i<laserCloudIn.points.size(); i++ )
    {
        double x_pos = laserCloudIn.points[i].x;
        double y_pos = laserCloudIn.points[i].y;
        double z_pos = laserCloudIn.points[i].z;
        double d_ = std::sqrt( x_pos * x_pos + y_pos*y_pos + z_pos * z_pos);
        if (d_ < d_th && laserCloudIn.points[i].z >VEHICLE_RANGE_Z_MIN && laserCloudIn.points[i].z<VEHICLE_RANGE_Z_MAX && laserCloudIn.points[i].y >VEHICLE_RANGE_Y_MIN && laserCloudIn.points[i].y <VEHICLE_RANGE_Y_MAX && laserCloudIn.points[i].x >VEHICLE_RANGE_X_MIN && laserCloudIn.points[i].x <VEHICLE_RANGE_X_MAX && x_pos>=0.11)
        {
            laserCloudVehicle->push_back(laserCloudIn.points[i]);
        }
    }
    
    
    sensor_msgs::PointCloud2 laserCloudSurround;
    pcl::toROSMsg(*laserCloudVehicle, laserCloudSurround);
    laserCloudSurround.header.stamp = laserCloudMsg->header.stamp;
    laserCloudSurround.header.frame_id = "/robot1_init";
    pubLaserVehicle.publish(laserCloudSurround);
    if (count_save ++ % 1 == 0 )
    {
        std::stringstream ss;
        std::string filename = PATH_PT;
        std::string filename_pt = PATH_PT;
        ss << count_save;
        std::string num = ss.str();
        filename.append(num);
         filename_pt.append(num);
        filename.append("_vfh.pcd");
        filename_pt.append("_pt.pcd");
        pcl::io::savePCDFileASCII(filename_pt,*laserCloudVehicle);
        timeRecord << count_save << " " << laserCloudMsg->header.stamp <<std::endl;
        std::cout<<"size of car: "<<laserCloudVehicle->points.size()<<std::endl;    
    }
    

}





int main(int argc, char **argv)
{
    ros::init(argc, argv, "collect_datasets");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
    
    printf("scan line number %d \n", N_SCANS);
    
    ros::param::get("~vehicle_range_x_min",VEHICLE_RANGE_X_MIN); 
    ros::param::get("~vehicle_range_x_max", VEHICLE_RANGE_X_MAX);
    ros::param::get("~vehicle_range_y_min",VEHICLE_RANGE_Y_MIN); 
    ros::param::get("~vehicle_range_y_max", VEHICLE_RANGE_Y_MAX);
    ros::param::get("~vehicle_range_z_min",VEHICLE_RANGE_Z_MIN); 
    ros::param::get("~vehicle_range_z_max", VEHICLE_RANGE_Z_MAX);
    
    ros::param::get("~path_pt", PATH_PT);
    ros::param::get("~path_vfh", PATH_VFH);
    ros::param::get("~path_time", PATH_TIME);
    
    ros::param::get("~d_th", d_th);

    
    printf("x_min %f \n", VEHICLE_RANGE_X_MIN);
    printf("x_max %f \n", VEHICLE_RANGE_X_MAX);
    printf("y_min %f \n", VEHICLE_RANGE_Y_MIN);
    printf("y_max %f \n", VEHICLE_RANGE_Y_MAX);
    printf("z_min %f \n", VEHICLE_RANGE_Z_MIN);
    printf("z_max %f \n", VEHICLE_RANGE_Z_MAX);
    
    printf("d_th %f \n", d_th);
    
    
    printf("Save pcd/vfh files to %s \n", PATH_PT.c_str());
    printf("Save vfh values to %s \n", PATH_VFH.c_str());
    printf("Save time stamps to %s \n", PATH_TIME.c_str());
    

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/lidar_cloud_origin", 100, laserCloudHandler);
    pubLaserVehicle = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_vehicle", 100);


    ros::spin();

    return 0;
}
