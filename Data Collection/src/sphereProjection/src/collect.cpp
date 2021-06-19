#include <cmath>
#include <vector>
#include <string>
#include "sphere_projection/common.h"
#include "sphere_projection/tic_toc.h"
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

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui.hpp>
#include<mutex>
#include<math.h>

#include <eigen3/Eigen/Dense>

#define PI 3.1415926


using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
double MINIMUM_RANGE = 0.1; 



int count_save = 0;


std::mutex mBuf;
std::mutex mPose;


double width = 360;
double height = 16;

int flag=0;

pcl::search::KdTree<pcl::PointXYZI>::Ptr tree_normal (new pcl::search::KdTree<pcl::PointXYZI> ());

double theta_max = -1;
double theta_min = 10000;



pcl::PointCloud<pcl::Normal>::Ptr calNormalMap(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudVehicle)
{
  mBuf.lock();
	pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
	ne.setInputCloud(laserCloudVehicle);
	ne.setSearchMethod (tree_normal);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setKSearch (5);
	ne.compute (*cloud_normals);
  mBuf.unlock();
  return cloud_normals;
}


std::string PATH_PT = "/media/mjy/Samsung_T5/linux/IV/work/4.11/2/vfh_related/vfh/";
std::string VERTEX_IMG_ROOT = "/media/mjy/Samsung_T5/linux/ICUS2021/vertex/test/";



int main(int argc, char **argv)
{
    ros::init(argc, argv, "sphere");
    ros::NodeHandle nh;

    ros::param::get("~path_pt", PATH_PT);
    ros::param::get("~vertex_img_root", VERTEX_IMG_ROOT);
    ros::param::get("~img_height", height);
    ros::param::get("~img_width", width);
    
    printf("Source from %s \n", PATH_PT.c_str());
    printf("Save to %s \n", VERTEX_IMG_ROOT.c_str());
    printf("Shape of each image (%lf, %lf) \n", (height, width));

    typedef pcl::PointXYZI PointT;
    pcl::PointCloud<PointT>::Ptr cloud_show (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_right (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_all (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_used_left (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_used_right (new pcl::PointCloud<PointT>);
    
    pcl::PointCloud<PointT>::Ptr cloud_data (new pcl::PointCloud<PointT>);

    for (int k=0; k<40000; k++)
    {
        std::stringstream ss;
        std::string filename_pt = PATH_PT;
        ss << k+1;
        std::string num = ss.str();
        filename_pt.append(num);
        filename_pt.append("_pt.pcd");
        
        std::cout<<"Saving "<<k<<" images from "<<  filename_pt<<std::endl;

        if (pcl::io::loadPCDFile<PointT> (filename_pt, *cloud_show) == -1)
            {
            //* load the file
            PCL_ERROR ("Couldn't read PCD file \n");
            return (-1);
            }
        
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = calNormalMap(cloud_show);

        // 16 * 720
        cv::Mat vertex_map( height,width, CV_8UC1, cv::Scalar(0));
        Eigen::MatrixXd vertex_map_n0 =  Eigen::MatrixXd::Zero(height,width);
        Eigen::MatrixXd vertex_map_n1 =  Eigen::MatrixXd::Zero(height,width);
        Eigen::MatrixXd vertex_map_n2 =  Eigen::MatrixXd::Zero(height,width);
        Eigen::MatrixXd depth =  Eigen::MatrixXd::Zero(height,width);
        Eigen::MatrixXd intensity_map =  Eigen::MatrixXd::Zero(height,width);


        for (int i=0; i<cloud_show->points.size(); i++)
        {
            double x = cloud_show->points[i].x;
            double y = cloud_show->points[i].y;
            double z = cloud_show->points[i].z;
            double r = std::sqrt(x*x+y*y+z*z);
            
            if (r < 0.5)
            {
                continue;
            }            

            double theta = atan(y/x);

            if (theta < 0 && y > 0)
            {
                theta = theta + PI;
            }
            else if (theta > 0 && y < 0)
            {
                theta = theta - PI;
            }
            double u = 0.5 * (1 - theta/PI) * width;
            double v = 0.5 * (1 - (asin(z/r) + 15*PI/180) / (40*PI/180)) * height;
            
            if (x!=0 && y!=0 && z!=0)
            {
                int x_a =  int(u);
                int y_a =  int(v);
                if (x_a >=2000)  x_a = 0;

                uchar* data = vertex_map.ptr<uchar>(y_a);
                data[x_a] = 255;
                
                vertex_map_n0(y_a, x_a) = cloud_normals->points[i].normal_x ;
                vertex_map_n1(y_a, x_a) = cloud_normals->points[i].normal_y;
                vertex_map_n2(y_a, x_a) = cloud_normals->points[i].normal_z ;
                depth(y_a, x_a) = r;
                intensity_map(y_a, x_a)= cloud_show->points[i].intensity;
            }
            
        }
        
            std::string filename_save_intensity = VERTEX_IMG_ROOT;
            filename_save_intensity.append("intensity/");
            filename_save_intensity.append(num);
            filename_save_intensity.append(".txt");
            std::ofstream intensity_record(filename_save_intensity);
            for (int m=0; m<height; m++)
            {
                for (int n=0; n<width; n++)
                {
                    
                    intensity_record <<intensity_map(m,n) << " ";
                }
                intensity_record<<std::endl;
            }
            
            std::string filename_save_n0 = VERTEX_IMG_ROOT;
            filename_save_n0.append("vertex_img_n0/");
            filename_save_n0.append(num);
            filename_save_n0.append(".txt");
            std::ofstream vertex_map_n0_record(filename_save_n0);
            for (int m=0; m<height; m++)
            {
                for (int n=0; n<width; n++)
                {
                    
                    vertex_map_n0_record <<vertex_map_n0(m,n) << " ";
                }
                vertex_map_n0_record<<std::endl;
            }
            

            
            std::string filename_save_n1 = VERTEX_IMG_ROOT;
            filename_save_n1.append("vertex_img_n1/");
            filename_save_n1.append(num);
            filename_save_n1.append(".txt");
            std::ofstream vertex_map_n1_record(filename_save_n1);
            for (int m=0; m<height; m++)
            {
                for (int n=0; n<width; n++)
                {
                    vertex_map_n1_record <<vertex_map_n1(m,n) << " ";
                }
                vertex_map_n1_record<<std::endl;
            }
            
            
            
            std::string filename_save_n2 = VERTEX_IMG_ROOT;
            filename_save_n2.append("vertex_img_n2/");
            filename_save_n2.append(num);
            filename_save_n2.append(".txt");
            std::ofstream vertex_map_n2_record(filename_save_n2);
            for (int m=0; m<height; m++)
            {
                for (int n=0; n<width; n++)
                {
                    vertex_map_n2_record <<vertex_map_n2(m,n) << " ";
                }
                vertex_map_n2_record<<std::endl;
            }  
            
        
            std::string filename_save_depth = VERTEX_IMG_ROOT;
            filename_save_depth.append("depth/");
            filename_save_depth.append(num);
            filename_save_depth.append(".txt");
            std::ofstream vertex_map_depth_record(filename_save_depth);
            for (int m=0; m<height; m++)
            {
                for (int n=0; n<width; n++)
                {
                    vertex_map_depth_record <<depth(m,n) << " ";
                }
                vertex_map_depth_record<<std::endl;
            }
            
            
    }
//     ros::spin();

    return 0;
}
