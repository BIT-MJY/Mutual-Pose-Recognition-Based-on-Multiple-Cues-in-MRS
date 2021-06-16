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
### Robot B Poses for  Mutual Poses
Here ALOAM is used by Robot B and the accurate poses from the topic ```/aft_mapped_to_init``` is recorded with [capturepose node].


Coming soon......


Developed by Junyi Ma, Jinyi Xu.




