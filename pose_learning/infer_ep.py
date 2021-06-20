# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: Infer with error propagation.

import net_rs
import net_vlp
import generate_data as gd
import tensorflow as tf
import numpy as np
import os
import yaml
import logging
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import visualization.show_infer_ep as sie


tfk = tf.keras
Input = tfk.layers.Input
Conv2D = tfk.layers.Conv2D
Dense = tfk.layers.Dense 
Flatten = tfk.layers.Flatten
Reshape = tfk.layers.Reshape
Lambda= tfk.layers.Lambda
Dropout= tfk.layers.Dropout
Model = tfk.models.Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
gpus= tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_soft_device_placement = False 
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus), 'Logical gpus')

tf.compat.v1.disable_eager_execution()

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

K.clear_session()
# 读取配置文件 ================================================================================================================

logger.info("     loading configure ...")

config_filename = "./config.yaml"
config = yaml.load(open(config_filename))
height = config['global_setting']['height']
width = config['global_setting']['width']
use_depth = config['train']['use_depth']
use_intensity = config['train']['use_intensity']
use_normals = config['train']['use_normals']
use_semantic = config['train']['use_semantic']

initial_lr = config['train']['initial_lr']
lr_alpha = config['train']['lr_alpha']
no_channels = np.sum(use_depth) + np.sum(use_intensity) + np.sum(use_normals) *3
use_vlp = config['global_setting']['use_vlp']
weights_filename = config['train']['weights_filename']
image_path = config['train']['image_path']
all_files_depth_dst = config['txt2npy']['all_files_depth_dst']
alinged_poses_root = config['align_images_poses']['alinged_poses_save_dst']
batch_size = config['infer_ep']['batch_size']   # batch size修改为自己配置
start_test_num = config['infer_ep']['start_test_num']
end_test_num = config['infer_ep']['end_test_num']
ep_version_old = config['infer_ep']['ep_version_old']
if ep_version_old:
  import error_propagation_old as ep  # 建议使用旧版本
else:
  import error_propagation_new as ep   


logger.info("     building model ...")
input_shape = (height, width, no_channels)
if use_vlp:
  logger.info("     using VLP-16 ...")
  model = net_vlp.generateNetwork(input_shape)
else:
  logger.info("     using RS-80 ...")
  model = net_rs.generateNetwork(input_shape)

optimizer = tfk.optimizers.Adagrad(lr=initial_lr)

model.compile(loss="mean_squared_error", optimizer=optimizer)
model.summary()


test_num = end_test_num - start_test_num 

test_data = np.zeros((test_num,height,width,no_channels))
for i in tqdm(range(0,test_num)):
    channelidx = 0
    if use_depth:
      f = os.path.join(image_path, 'depth', str(start_test_num+i+1) + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
        raise Exception('Could not read depth image %s' % f)
      
      test_data[i, :, :, channelidx] = img1.reshape(height, width)
      channelidx += 1
    
    if use_normals:
      f = os.path.join(image_path, 'normal',str(start_test_num+i+1) + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
          raise Exception('Could not read normal image %s' % f)
        
      test_data[i, :, :, channelidx:channelidx+3] = img1
      channelidx += 3
    
    # if use_semantic:
    #   # normal map == channel 1..3
    #   f = os.path.join(image_path, 'semantic', str(i+1)+ '.npy')
    #   try:
    #     img1 = np.load(f)
    #   except IOError:  
    #       raise Exception('Could not read semantic image %s' % f)
        
    #   test_data[i, :, :, channelidx] = img1
    #   channelidx += 1

    if use_intensity:
      f = os.path.join(image_path, 'intensity', str(start_test_num+i+1)+ '.npy')
      try:
        img1 = np.load(f)
      except IOError:
        # Try to read format .npz
        f = os.path.join(image_path,  'intensity', str(start_test_num+i+1)+ '.npz')
        img1 = np.load(f)
      img1 = img1/255
      test_data[i, :, :, channelidx] = img1.reshape(height, width)
      channelidx += 1
    

model.load_weights("log/cp/pretrained_weights.h5")

poses = np.load(alinged_poses_root+"aligned_poses.npy")
y_true = poses[start_test_num:end_test_num,:]

ep_model = ep.EP(model)
ep_model_used = ep_model.create_EP_Model()

f = K.function( [ep_model.model.layers[0].input, K.symbolic_learning_phase()], [ep_model.model.output]  )


ep_preds_all = np.zeros((end_test_num-start_test_num, 7))
var_all = np.zeros((end_test_num-start_test_num, 7))

iter_all = int( (end_test_num-start_test_num)/batch_size )

for iter in tqdm(range(iter_all)):
  index_start = iter * batch_size
  index_end = (iter+1) * batch_size



  if index_start + start_test_num >= end_test_num:
    break

  if index_end + start_test_num >= end_test_num:
    index_end = end_test_num

  test_fragment = test_data[index_start:index_end,:,:,:]
  ep_preds, var = f([test_fragment, False])[0]


  ep_preds_all[iter * batch_size:(iter+1) * batch_size, :] = ep_preds
  var_all[iter * batch_size:(iter+1) * batch_size, :] = var


std_all = np.sqrt(var_all)

sie.show_pred_ep(ep_preds_all, var_all, y_true, [0,1,2],False)