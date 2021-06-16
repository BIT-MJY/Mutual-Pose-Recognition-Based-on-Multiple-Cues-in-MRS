# Mutual-Pose-Recognition-Based-on-Multiple-Cues-in-MRS
Source Code for ICUS 2021: Mutual Pose Recognition Based on Multiple Cues in Multi-robot System.

## Overview
Here is the method to utilize the multiple cues including depth maps, normal maps, remission maps, and semantic maps to recognize the mutual poses of pair-wise robots. We first collect the corresponding laser points attached to robot teammate, and then project these points to images fed to the devised CNN. The devised CNN outputs 6-DOF mutual poses. At the same time, error propagation is implemented to capture the uncertainty of the estimated mutual poses. Uncertain, i.e., "over confident" mutual poses are filtered out and not utilized by the following tasks, such as local map merging in multi-robot SLAM (MR-SLAM).

* Data Collection
* Spherical Projection
* CNN Building and Training

## Related Work
* Spherical Projection is implemented along the lines of [OverlapNet](https://github.com/BIT-MJY/OverlapNet_for_TF2).

## Data Collection
To train a CNN to regress out 6-DOF mutual poses, mapping from **point clouds attached to the robot teammate** to **mutual poses** should be collected.


Coming soon......


Developed by Junyi Ma, Jinyi Xu.
