# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: Transform the file formats saving multiple cues. (.txt to .npy.)

import numpy as np
import os
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import yaml
import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import warnings
warnings.filterwarnings("ignore")

config_filename = "./config.yaml"
config = yaml.load(open(config_filename))


use_depth = config['txt2npy']['use_depth']
use_intensity = config['txt2npy']['use_intensity']
use_normals = config['txt2npy']['use_normals']
show_data = config['txt2npy']['show_data']
show_index = config['txt2npy']['show_index']
save_image = config['txt2npy']['save_image']


all_files_depth_root = config['txt2npy']['all_files_depth_root']
all_files_depth_dst = config['txt2npy']['all_files_depth_dst']
all_files_intensity_root = config['txt2npy']['all_files_intensity_root']
all_files_intensity_dst = config['txt2npy']['all_files_intensity_dst']
all_files_normal0_root = config['txt2npy']['all_files_normal0_root']
all_files_normal1_root = config['txt2npy']['all_files_normal1_root']
all_files_normal2_root = config['txt2npy']['all_files_normal2_root']
all_files_normal_dst = config['txt2npy']['all_files_normal_dst']

height = config['global_setting']['height']
width = config['global_setting']['width']




if use_depth: # =================================================================================================================

    all_files_depth = os.listdir(all_files_depth_root)

    logger.info("    Using %d depth images", len(all_files_depth))

    depth_all = np.zeros((len(all_files_depth), height,width,1))

    k = 0

    for file in all_files_depth:


        depth = np.zeros((height,width,1))

        file_depth = all_files_depth_root + file

        flag = 0
        with open(file_depth, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                for i in range(width):
                    depth[flag,i,:] = float(line_list[i]) 
                    depth_all[k,flag,i,:] = float(line_list[i]) 
                flag = flag + 1

        dst_folder = all_files_depth_dst
        try:
            os.stat(dst_folder)
        except:
            print('creating new depth folder: ', dst_folder)
            os.mkdir(dst_folder)
        dst_save_depth = dst_folder + file.split('.')[0] 
        np.save(dst_save_depth, depth)
        print("Saving depth image to "+dst_save_depth+"...")
        # print(np.unique(depth))

        # if k==0:
        #     pri

        k = k + 1

if use_intensity: # =================================================================================================================

    all_files_intensity = os.listdir(all_files_intensity_root)

    logger.info("    Using %d intensity images", len(all_files_intensity))

    for file in all_files_intensity:

        intensity = np.zeros((height,width,1))

        file_intensity = all_files_intensity_root + file

        flag = 0
        with open(file_intensity, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                for i in range(width):
                    intensity[flag,i,:] = float(line_list[i]) 
                flag = flag + 1

        dst_folder = all_files_intensity_dst
        try:
            os.stat(dst_folder)
        except:
            print('creating new intensity folder: ', dst_folder)
            os.mkdir(dst_folder)
        dst_save_intensity = dst_folder + file.split('.')[0] 
        np.save(dst_save_intensity, intensity)
        print("Saving intensity image to "+dst_save_intensity+"...")


if use_normals:   # =================================================================================================================

    all_files_n0 = os.listdir(all_files_normal0_root)
    all_files_n1 = os.listdir(all_files_normal1_root)
    all_files_n2 = os.listdir(all_files_normal2_root)

    if len(all_files_n0) != len(all_files_n1) or \
        len(all_files_n0) != len(all_files_n2) or \
            len(all_files_n1) != len(all_files_n2):
            # print("n0 n1 n2 are not totally equal.......")
            raise Exception("n0 n1 n2 are not totally equal.......")

    logger.info("    Using %d normal images", len(all_files_n0))


    for num in range( len(all_files_n0)  ):
        normals = np.zeros((height,width,3))
        file = os.listdir(all_files_normal0_root)[num]

        file_n0 = all_files_normal0_root + file
        flag = 0
        with open(file_n0, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                for i in range(width):
                    normals[flag,i,0] = float(line_list[i]) 
                flag = flag + 1

        file_n1 = all_files_normal1_root + file
        flag = 0
        with open(file_n1, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                for i in range(width):
                    normals[flag,i,1] = float(line_list[i]) 
                flag = flag + 1
        
        file_n2 = all_files_normal2_root + file
        flag = 0
        with open(file_n2, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                for i in range(width):
                    normals[flag,i,2] = float(line_list[i]) 
                flag = flag + 1

        dst_folder = all_files_normal_dst
        try:
            os.stat(dst_folder)
        except:
            print('creating new normal folder: ', dst_folder)
            os.mkdir(dst_folder)
        dst_save_normals = dst_folder + file.split('.')[0] 
        np.save(dst_save_normals, normals)
        print("Saving normal image to "+dst_save_normals+"...")


if show_data:

    plt.figure(figsize=(0,0))
    plt.figure(figsize=(40,40))
    plt.subplot(3,1,1)
    plt.title('depth map')
    plt.imshow(np.load("data/depth/" + str(show_index) + ".npy"))

    plt.subplot(3,1,2)
    plt.title('intensity map')
    plt.imshow(np.load("data/intensity/" + str(show_index) + ".npy"))

    plt.subplot(3,1,3)
    plt.title('normal map')
    plt.imshow(np.load("data/normal/" + str(show_index) + ".npy") * 255)

    plt.clf()

    if save_image:
        plt.figure(figsize=(30,5))
        img = plt.imshow(np.load("data/depth/" + str(show_index) + ".npy"))
        plt.savefig("img/depth_"+str(show_index)+".eps")
        plt.clf()
        plt.figure(figsize=(30,5))
        img = plt.imshow(np.load("data/intensity/" + str(show_index) + ".npy"))
        plt.savefig("img/intensity_"+str(show_index)+".eps")
        plt.clf()
        plt.figure(figsize=(30,5))
        img = plt.imshow(np.load("data/normal/" + str(show_index) + ".npy"))
        plt.savefig("img/normal_"+str(show_index)+".eps")
        plt.clf()

    print("Saving Successfully!")

    if not save_image:
        plt.show()

