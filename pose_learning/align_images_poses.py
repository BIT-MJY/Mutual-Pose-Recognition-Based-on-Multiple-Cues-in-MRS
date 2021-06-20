# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: Align the time of laser information with the time of gt poses.

import numpy as np
import sys
import os
import yaml
from tqdm import tqdm


def alignImagesPoses(watch_num, align_num, poses_root, times_root):

    # x= np.arange(1,34)

    count = 0

    y_train = np.zeros((25000,7))

    y_train_aligned = np.zeros((watch_num,7))

    X_time = []
    y_time = []


    with open(poses_root+"truePose.txt", "r") as f:
        for line in f.readlines():
            # watch how many data
            # if (count == watch_num):   # watch how many data
            #     break
            line_sp = line.split(" ")
            # print(line_sp)
            flag = 0
            pos = 0
            for num in line_sp:
            # print(num)
                if flag==0:
                    flag = flag + 1
                    continue
            # if flag == 7:
            #   num = num[1:-2]
                y_train[count, pos]  = float(num)
                pos = pos + 1
                if pos == 7:
                    break
            count = count + 1
            flag = flag + 1




    start = 0
    with open(times_root+"time.txt", "r") as f:
        for line in f.readlines():
            # if (start <11000):
            #   start = start + 1
            #   continue
            X_time.append(float(line.split(" ")[1][0:-2]))
        print(len(X_time))
    with open(poses_root+"timePose.txt", "r") as f:
        for line in f.readlines():
            y_time.append(float(line.split(" ")[1][0:-2]))
        print(len(y_time))
    # print(y_time)

    for i in tqdm(range(watch_num)):
        for k in range(len(y_time)-1):
            if (X_time[i] > y_time[k]) and (X_time[i] < y_time[k+1]):
                s = (X_time[i] - y_time[k])/(y_time[k+1] - y_time[k])
                # q1 = np.quaternion(y_train[k,4],y_train[k,5],y_train[k,6],y_train[k,7])
                # q2 = np.quaternion(y_train[k+1,4],y_train[k+1,5],y_train[k+1,6],y_train[k+1,7])
                # quaternion.slerp_evaluate(q1,q2,s)
                y_train_aligned[i,:] = y_train[k,:]+s*(y_train[k+1,:] - y_train[k,:])
                y_train_aligned[i,3:] = y_train_aligned[i,3:] / np.sqrt(y_train_aligned[i,3]**2 + y_train_aligned[i,4]**2 + y_train_aligned[i,5]**2 + y_train_aligned[i,6]**2)
                
    
    print("The last pose is interpolated by:")
    print(y_train[k,:])
    print(y_train[k+1,:])

    return y_train_aligned

if __name__ == "__main__":

    config_filename = "./config.yaml"
    config = yaml.load(open(config_filename))
    use_depth = config['train']['use_depth']
    use_intensity = config['train']['use_intensity']
    use_normals = config['train']['use_normals']
    poses_data_root = config['align_images_poses']['poses_data_root']
    raw_data_root = config['align_images_poses']['raw_data_root']
    alinged_poses_save_dst = config['align_images_poses']['alinged_poses_save_dst']
    
    no_files = 0

    if use_depth:
        all_files_depth_dst = config['txt2npy']['all_files_depth_dst']
        no_files = len(os.listdir(all_files_depth_dst))
    if use_intensity:
        all_files_intensity_dst = config['txt2npy']['all_files_intensity_dst']
        no_files = len(os.listdir(all_files_intensity_dst))
    if use_normals:
        all_files_normal_dst = config['txt2npy']['all_files_normal_dst']
        no_files = len(os.listdir(all_files_normal_dst))

    align_num = 0
    with open(poses_data_root+"truePose.txt", "r") as f:
        for line in f.readlines():
            align_num = align_num + 1

    poses_aligned = alignImagesPoses(no_files, align_num, poses_data_root, raw_data_root)

    try:
        os.stat(alinged_poses_save_dst)
    except:
        print('creating new poses folder: ', alinged_poses_save_dst)
        os.mkdir(alinged_poses_save_dst)


    print("Saving aligned poses to", alinged_poses_save_dst+"aligned_poses.npy")
    np.save(alinged_poses_save_dst+"aligned_poses", poses_aligned)

    print("poses_aligned.shape: ", poses_aligned.shape)
    print("poses_aligned[-1]: ", poses_aligned[-1])

