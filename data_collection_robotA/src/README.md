# Source Code for Robot A

```
git clone https://github.com/BIT-MJY/Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS.git
cd Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS/data_collection_robotA
catkin_make
source devel/setup.bash
```

If you want to collect raw lidar data attached to robot B, please run
```
roslaunch cal_vfh collect_80.launch
```
If you want to transform the collected raw lidar data to vertex maps, please run
```
roslaunch sphere_projection sphere.launch
```
