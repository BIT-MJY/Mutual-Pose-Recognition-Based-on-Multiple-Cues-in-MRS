# Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS
Source Code for ICUS 2021: Mutual Pose Recognition Based on Multiple Cues in Multi-robot System.

## Overview
Here is the method to utilize the multiple cues including depth maps, normal maps, remission maps, and semantic maps to recognize the mutual poses of pair-wise robots. We first collect the corresponding laser points attached to robot teammate, and then project these points to images fed to the devised CNN. The devised CNN outputs 6-DOF mutual poses. At the same time, error propagation is implemented to capture the uncertainty of the estimated mutual poses. Uncertain, i.e., "over confident" mutual poses are filtered out and not utilized by the following tasks, such as local map merging in multi-robot SLAM (MR-SLAM).

* Data Collection
* Spherical Projection
* CNN Building and Training

## Related Work
* Spherical Projection is implemented along the lines of [OverlapNet](https://github.com/BIT-MJY/OverlapNet_for_TF2).
* Error propagation follows [Moment Propagation](https://github.com/kaibrach/Moment-Propagation.git).
* Recorded poses is from [ALOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).

## Data Collection
To train a CNN to regress out 6-DOF mutual poses, mapping from **point clouds attached to the robot teammate** to **mutual poses** should be collected. 
### Hardware and Software Preparation
* Two robots with lidar.
* Robot Operating System.
* Tensorflow 2.
### Deployment
Suppose you have two robots A and B with similar shapes. Then you collect laser points attached to robot B using the lidar of robot A. At the same time, poses of robot B should be recorded. Thus there are several pivotal problems:
#### Time Synchronization
Two robots are in the same WIFI:
* Robot A: 192.168.43.50
* Robot B: 192.168.43.100  
You can use the following command in the terminal (Robot A) to finish time synchronization between the two robots 
```
sudo ntpdate 192.168.43.100
```
#### Initial Mutual Pose
The initial mutual pose is utilized to calculate the real mutual poses. It can be realized by multi-lidar extrinsic calibration. We don't provide the source code for calibration, but it is still significant to validate the feasibility of our devised CNN (a gray box) and the following error propagation although "real" mutual poses are not "real".
#### Robot B Poses for Mutual Poses
Here [ALOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) is used by Robot B and the accurate poses from the topic ```/aft_mapped_to_init``` is recorded with [capturepose node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotB/src/capturePose/src). The results of ALOAM is thought of as ground truth. If you build the surrounding map in advance, scan-to-map can be utilized to collect GT poses. In addition, time stamps are recorded at the same time.  

Run ALOAM firstly for Robot B. Then, launch the [capturepose node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotB/src/capturePose/src) by
```
rosrun capturePose capturepose 
```
Params in [capturepose.cpp](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/blob/main/data_collection_robotB/src/capturePose/src/capturepose.cpp):
- **odomAftRecord**: The txt file recording the output poses of ALOAM.
- **timeAftRecord**: The txt file recording the time stamps pf output poses.

#### Collecting Point Clouds attached to robot B
Point Clouds attached to robot B should be segmented by RangeNet++ firstly. However, here we only provide the geometry cues and intensity cues for recognition. Thus, for the sake of convenience, we collect the laser points within the preset region of interest. In the ROI, there are no laser points except those attached to robot B. We provide the node [collect_only_vfh_node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotA/src/calVFH/src) for collecting the points and records the time stamps at the same time.

## Data Preprocessing
### Spherical Projection
Spherical projection is implemented based on the collected point clouds attached to robot B. The node [spherical_projection_node](https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/tree/main/data_collection_robotA/src/sphereProjection/src) is utilized to generate depth maps, intensity maps, and normal maps. These multiple maps are saved as txt.
### [txt to npy]
### [Interpolation]
### Structure
Pose_Learning
> raw_txt  
>> depth  
>>> 1.txt  
>>> 2.txt  
>>> ...  

>> intensity  
>>> 1.txt  
>>> 2.txt  
>>> ...


>> vertex_img_n0 
>>> 1.txt  
>>> 2.txt  
>>> ...  


>> vertex_img_n1 
>>> 1.txt  
>>> 2.txt  
>>> ...  
>> vertex_img_n2  
>>> 1.txt  
>>> 2.txt  
>>> ...   


>> poses  
>>> timePose.txt  
>>> truePose.txt  


>> time.txt  

> data  
>> The files in this folder are all generated automatically.  

> log  
>> cp  
>> weights  

> img  

## Training

## testing


Coming soon......


Developed by Junyi Ma, Jinyi Xu.




