# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: Infer without error propagation.


import net_rs
import net_vlp
import generate_data as gd
import sys
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

# import tensorflow.keras.backend as K

tfk = tf.keras
Input = tfk.layers.Input
Conv2D = tfk.layers.Conv2D
Dense = tfk.layers.Dense 
Flatten = tfk.layers.Flatten
Reshape = tfk.layers.Reshape
Lambda= tfk.layers.Lambda
Dropout= tfk.layers.Dropout
Model = tfk.models.Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
gpus= tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_soft_device_placement = False
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus), 'Logical gpus')

tf.config.experimental_run_functions_eagerly(True)  

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



logger.info("     loading configure ...")

config_filename = "./config.yaml"
config = yaml.load(open(config_filename))
height = config['global_setting']['height']
width = config['global_setting']['width']
use_depth = config['train']['use_depth']
use_intensity = config['train']['use_intensity']
use_normals = config['train']['use_normals']
use_semantic = config['train']['use_semantic']

no_channels = np.sum(use_depth) + np.sum(use_intensity) + np.sum(use_normals) *3
use_vlp = config['global_setting']['use_vlp']
weights_filename = config['train']['weights_filename']
image_path = config['train']['image_path']
all_files_depth_dst = config['txt2npy']['all_files_depth_dst']
alinged_poses_root = config['align_images_poses']['alinged_poses_save_dst']
val_rate = config['train']['val_rate']
no_epochs = config['train']['no_epochs']
batch_size = config['train']['batch_size']
initial_lr = config['train']['initial_lr']
lr_alpha = config['train']['lr_alpha']
test_developed = config['train']['test_developed']
log_path = config['train']['log_path']
watch = config['train']['watch']
watch_num = config['train']['watch_num']
use_generator = config['train']['use_generator']
start_test_num = config['infer']['start_test_num']
end_test_num = config['infer']['end_test_num']
weights_filename_cp = config['train']['weights_filename_cp']



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
    

    if use_intensity:
      f = os.path.join(image_path, 'intensity', str(start_test_num+i+1)+ '.npy')
      try:
        img1 = np.load(f)
      except IOError:
        f = os.path.join(image_path,  'intensity', str(start_test_num+i+1)+ '.npz')
        img1 = np.load(f)
      test_data[i, :, :, channelidx] = img1.reshape(height, width)
      channelidx += 1
    



model.load_weights(weights_filename_cp)
print(np.unique(test_data[:,:,:,0]).shape) 

preds = model.predict(test_data)

poses = np.load(alinged_poses_root+"aligned_poses.npy")
y_true = poses[start_test_num:end_test_num,:]

mean_error = np.mean(preds - y_true)


rmse = np.sqrt(np.sum( (preds - y_true)**2  ) / test_num /7)

print("-------mean_error:", mean_error)
print("-------rms_error:", rmse)


scores = model.evaluate(test_data,y_true, verbose=2, batch_size=32)
print(scores)


# plt.figure(figsize=(15,8))
# plt.plot(y_true[:,0])
# plt.plot(preds[:,0])
# plt.show()

# plt.figure(figsize=(15,8))
# plt.plot(y_true[:,1])
# plt.plot(preds[:,1])
# plt.show()

plt.figure(figsize=(15,8))
plt.plot(y_true[:,-1])
plt.plot(preds[:,-1])
plt.show()

