# Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS
Source Code for ICUS 2021: Mutual Pose Recognition Based on Multiple Cues in Multi-robot System.
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/system_overview.png" >

## Overview
Here is the method to utilize the multiple cues, including depth maps, normal maps, remission maps, and semantic maps, to recognize the mutual poses of pair-wise robots. We first collect the corresponding laser points attached to the robot teammate, and then project these points to images fed to the devised CNN. The devised CNN outputs 6-DOF mutual poses. At the same time, error propagation is implemented to capture the uncertainty of the estimated mutual poses. Uncertain, i.e., "overconfident" mutual poses are filtered out and not utilized by the following tasks, such as local map merging in multi-robot SLAM (MR-SLAM).

* [Data Collection](#data-collection)
* [Data Preprocessing](#data-preprocessing)
* [CNN Training and Testing](#training-and-testing)

We only provide offline operation guidance for mutual pose recognition. 

## Related Work
* Spherical Projection is implemented along the lines of [OverlapNet](https://github.com/BIT-MJY/OverlapNet_for_TF2).
* Error propagation follows [Moment Propagation](https://github.com/kaibrach/Moment-Propagation.git).
   * Kai Brach, Beate Sick, and Oliver Durr. Single shot MC dropout approximation. arXiv preprint arXiv:2007.03293, 2020.
* Recorded poses is from [ALOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).

## Data Collection
To train a CNN to regress out 6-DOF mutual poses, mappings from **point clouds attached to the robot teammate** to **mutual poses** should be collected. 
### Hardware and Software Preparation
* Two robots with lidar.
* Robot Operating System.
* Tensorflow 2.
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/robots.PNG" width="270" height="200">

### Deployment
Suppose we have two robots A and B with similar shapes. Then we collect laser points attached to robot B using the lidar of robot A. At the same time, the poses of robot B should be recorded. Thus there are several pivotal problems:
#### Time Synchronization
Two robots are connected to the same WIFI:
* Robot A: 192.168.43.50
* Robot B: 192.168.43.100  
We can use the following command in the terminal (Robot A) to finish time synchronization between the two robots.
```
sudo ntpdate 192.168.43.100
```
#### Initial Mutual Pose
The initial mutual pose is utilized to calculate the real mutual poses. It can be realized by multi-lidar extrinsic calibration. We don't provide the source code for calibration, but it is still significant to validate the feasibility of our devised CNN (a gray box) and the following error propagation although "real" mutual poses are not "real".
#### Robot B Poses for Mutual Poses
Here [ALOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) is used by robot B and the accurate poses from the topic ```/aft_mapped_to_init``` is recorded with [capturepose node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotB/src/capturePose/src). The results of ALOAM is thought of as ground truth. It is better If we build the dense surrounding map in advance. In this case, Scan-to-map can be utilized to collect more accurate GT poses. In addition, timestamps are recorded at the same time.  

Run ALOAM firstly for Robot B. Then, launch the [capturepose node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotB/src/capturePose/src) by
```
rosrun capturePose capturepose 
```
Params in [capturepose.cpp](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/data_collection_robotB/src/capturePose/src/capturepose.cpp):
- **odomAftRecord**: The txt file recording the output poses of ALOAM.
- **timeAftRecord**: The txt file recording the time stamps pf output poses.

#### Collecting Point Clouds attached to robot B
Point Clouds attached to robot B should be segmented by RangeNet++ firstly. However, here we only provide the geometry cues and intensity cues for recognition. Thus, we collect the laser points within the preset region of interest for the sake of convenience. In the ROI, there are no laser points except those attached to robot B. We provide the node [collect_only_vfh_node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotA/src/calVFH/src) for collecting the points and records the timestamps at the same time. Run as
```
roslaunch cal_vfh collect_80.launch
```
Params in [collect_80.launch](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/data_collection_robotA/src/calVFH/launch/collect_80.launch):
- **capture_x/y_min/max**: ROI for robot B. In fact, the task to segment the region of the robot teammate is completed by RangeNet++. Here we only provide the preliminary version, so please ensure that the preset ROI only contains the robot B. The point clouds within this region are collected and saved to **path_pt_for_save**.
- **path_pt_for_save**: The path for saving raw point clouds.
- **path_time_for_save**: The path for saving the timestamps of each collected point clouds.


## Data Preprocessing
### Spherical Projection
Spherical projection is implemented based on the collected point clouds attached to robot B. The node [spherical_projection_node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotA/src/sphereProjection/src) is utilized to generate depth maps, intensity maps, and normal maps. These multiple maps are saved as .txt. Please run as 
```
roslaunch sphere_projection sphere.launch
```
Params in [sphere.launch](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/data_collection_robotA/src/sphereProjection/launch/sphere.launch):
- **path_pt_source**: The folder saving raw lidar data .pcd from "Collecting Point Clouds attached to robot B". It is same with **path_pt_for_save**.
- **vertex_root**: The folder to save various types of maps. Six folders should be in **vertex_root** including depth, intensity, vertex_img, vertex_img_n0, vertex_img_n1, vertex_img_n2.
- **image_height**: The height of the vertex maps.
- **image_width**: The width of the vertex maps.

### [txt to npy](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/txt2npy.py)
Please put .txt files from **vertex_root** in the [right place](#structure). Then run
```
Python3 txt2npy.py
```
Params for txt2npy.py are set in [config.xml](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/config.yaml).
- **use_depth**: Whether to transform raw_txt/depth/*.txt to data/depth/*.txt.
- **use_intensity**: Whether to transform raw_txt/intensity/*.txt to data/intensity/*.txt.
- **use_normals**: Whether to transform raw_txt/normal/*.txt to data/normal/*.txt.
- **show_data**: Whether to show loaded images.
- **show_index**: The index of the image to be shown if **show_data** is true.
- **save_image**: Whether to save the shown image.

- **all_files_depth_root**: "raw_txt/depth/"
- **all_files_depth_dst**: "data/depth/"

- **all_files_intensity_root**: "raw_txt/intensity/"
- **all_files_intensity_dst**: "data/intensity/"

- **all_files_normal0_root**: "raw_txt/vertex_img_n0/"
- **all_files_normal1_root**: "raw_txt/vertex_img_n1/"
- **all_files_normal2_root**: "raw_txt/vertex_img_n2/"
- **all_files_normal_dst**: "data/normal/"

### [Interpolation](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/align_images_poses.py)
To align the time of laser information with the time of gt poses, please run
```
Python3 align_images_poses.py
```
- **poses_data_root**: "raw_txt/poses/"
- **raw_data_root**: "raw_txt/"
- **alinged_poses_save_dst**: "data/poses/"
Finally, there is a file called **aligned_poses.npy** saving the aligned poses for vertex images under **alinged_poses_save_dst**.

### Structure
├─pose_learning  
  >>├─raw_txt  
    >>>├─depth  
      ├─1.txt  
      ├─2.txt  
      ├─...  
    ├─depth intensity  
      ├─1.txt  
      ├─2.txt  
      ├─...
    ├─vertex_img_n0 
      ├─1.txt  
      ├─2.txt  
      ├─...
    ├─vertex_img_n1 
      ├─1.txt  
      ├─2.txt  
      ├─... 
    ├─vertex_img_n2  
      ├─1.txt  
      ├─2.txt  
      ├─...
    ├─poses  
      ├─timePose.txt  
      ├─truePose.txt  
    ├─time.txt  
  ├─data  
      The files under this folder are all generated automatically.  
  ├─log  
    ├─cp  
    ├─weights  
  ├─img  
  ├─visualization

## Training and Testing
To train [the devised CNN](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/net_rs.py), please run
```
Python3 train.py
```
Params for train.py are set in [config.xml](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/config.yaml). You can choose the combinations of different cues to train the corresponding network.  
To test without error propagation, run
```
Python3 infer.py
```
You can specify the start index and the end index in [config.xml](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/pose_learning/config.yaml) for test. 
To test with error propagation, run
```
Python3 infer_ep.py
```
<div align=center>
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/tx.png" width="200">
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/ty.png" width="200">
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/tx2.png" width="200">
<img src="https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/img/ty2.png" width="200">

</div>

## Dataset
Please contact 3120200365@bit.edu.cn for the images-poses dataset.

## Authors
Developed by Junyi Ma.

## Acknowledgment
I would like to thank Kai Brach and Oliver Dürr for the source code about error propagation, and thank Jingyi Xu for helping with the experiments.






