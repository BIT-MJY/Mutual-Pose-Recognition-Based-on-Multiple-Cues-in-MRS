# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: Train a devised CNN.


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
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.keras import backend as K

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
                                                           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    
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
no_channels = np.sum(use_depth) + np.sum(use_intensity) + np.sum(use_normals) *3
use_vlp = config['global_setting']['use_vlp']
weights_filename = config['train']['weights_filename']
weights_filename_cp = config['train']['weights_filename_cp']
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
shuffle_data= config['train']['shuffle_data']


logger.info("----------height: %d", height)
logger.info("----------width: %d", width)
logger.info("----------use_depth: %d", np.sum(use_depth))
logger.info("----------use_intensity: %d", np.sum(use_intensity))
logger.info("----------use_normals: %d", np.sum(use_normals))
logger.info("----------no_channels: %d", no_channels)
logger.info("----------image_path: %s", image_path)
logger.info("----------val_rate: %s", val_rate)
logger.info("----------no_epochs: %s", no_epochs)
logger.info("----------batch_size: %s", batch_size)
logger.info("----------initial_lr: %s", initial_lr)
logger.info("----------lr_alpha: %s", lr_alpha)


def learning_rate_schedule(initial_lr=1e-3, alpha=0.99):

  def scheduler(epoch):
    if epoch < 1:
      # return initial_lr * 0.1
      return initial_lr 
    else:
      print("learning rate decay to ", initial_lr * np.power(alpha, epoch - 1.0))
      return initial_lr * np.power(alpha, epoch - 1.0)
  return tfk.callbacks.LearningRateScheduler(scheduler)

class LossHistory(tfk.callbacks.Callback):
  """ Small class for callback to record loss after each batch.
  """
  
  def on_train_begin(self, logs={}):
    self.losses = []
  
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))



def coeff_determination(y_true, y_pred):
  SS_res =  K.sum(K.square( y_true-y_pred ))
  SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
  return ( 1 - SS_res/(SS_tot + K.epsilon()) )


logger.info("     building model ...")
input_shape = (height, width, no_channels)
if use_vlp:
  logger.info("     using VLP-16 ...")
  model = net_vlp.generateNetwork(input_shape)
else:
  logger.info("     using RS-80 ...")
  model = net_rs.generateNetwork(input_shape)

optimizer = tfk.optimizers.Adagrad(lr=initial_lr)

model.compile(loss="mean_squared_error", optimizer='adam',metrics=[coeff_determination])
model.summary()

# pre-trained:
if len(weights_filename) > 0:
  if os.path.exists(weights_filename):
    model.load_weights(weights_filename)
    logger.info("     Beginning to fine tune for "+weights_filename)


imgfilenames1 = [ file.split('.')[0] for file in  os.listdir(all_files_depth_dst) ]
imgfilenames1.sort(key= lambda x:int(x))

if test_developed:
  poses = np.ones((30,1))
else:
  poses = np.load(alinged_poses_root+"aligned_poses.npy")

# check
if poses.shape[0]!=len(imgfilenames1):
  logger.error("Poses are not equal to images......")
  raise Exception('Shutdown')


if watch:
  poses = poses[:watch_num, :]
  imgfilenames1 = imgfilenames1[:watch_num]



if shuffle_data:
  shuffled_idx=np.random.permutation(poses.shape[0])
else:
  shuffled_idx = range(poses.shape[0])

no_val = int( poses.shape[0]*val_rate )
shuffled_idx_val = shuffled_idx[0:no_val]
shuffled_idx_train = shuffled_idx[no_val:]


poses_train = poses[shuffled_idx_train,:]
imgfilenames1_train = (np.array(imgfilenames1)[shuffled_idx_train]).tolist()
poses_val = poses[shuffled_idx_val,:]
imgfilenames1_val = (np.array(imgfilenames1)[shuffled_idx_val]).tolist()

if use_generator:
  train_generator = gd.ImagePoseMapping(image_path, imgfilenames1_train, poses_train,
                                  batch_size, height, width, no_channels=no_channels,
                                  use_depth=use_depth, use_normals=use_normals, use_semantic=False,
                                  use_intensity=use_intensity)
  validation_generator = gd.ImagePoseMapping(image_path, imgfilenames1_val, poses_val,
                                  batch_size, height, width, no_channels=no_channels,
                                  use_depth=use_depth, use_normals=use_normals, use_semantic=False,
                                  use_intensity=use_intensity)

  batchLossHistory = LossHistory()

  print("training！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")

  learning_rate = learning_rate_schedule(initial_lr=initial_lr, alpha=lr_alpha)
  time = datetime.datetime.now().time()
  log_dir = "%s/%d_%d/" % (log_path, time.hour, time.minute)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  writer = tf.summary.create_file_writer(log_dir)

  for epoch in range(0, no_epochs):
      history = model.fit_generator(train_generator,
                                  initial_epoch=epoch, epochs=epoch + 1,
                                  callbacks=[batchLossHistory, learning_rate],
                                  # callbacks=[batchLossHistory],
                                  max_queue_size=10, workers=4,
                                  use_multiprocessing=False
                                  )

      model_outputs = model.predict_generator(validation_generator, max_queue_size=10,
                                            workers=4, use_multiprocessing=False, verbose=1)
      diffs = abs(np.squeeze(model_outputs[0])-poses_val)
      mean_diff = np.mean(diffs)
      mean_square_error = np.mean(diffs*diffs)
      rms_error = np.sqrt(mean_square_error)
      max_error = np.max(diffs)

      print("-------mean_diff: ", mean_diff)
      print("-------mean_square_error: ", mean_square_error)
      print("-------rms_error: ", rms_error)
      print("-------max_error: ", max_error)


      # Saving weights 
      interval = 300  # save times
      save_point = [point*int(no_epochs/interval) for point in range(300)]
      if (np.sum(np.array(save_point)==epoch))==1:
        if len(weights_filename) > 0:
          logger.info("     saving model weights ...")
          model.save(weights_filename)
          print("Saving model to ", weights_filename)


      
      if epoch==no_epochs-1:
          if len(weights_filename) > 0:
            logger.info("     Saving the last model weights ...")
            model.save(weights_filename)
            logger.info("     Tranining is over")
else:

  x_train = np.zeros((poses.shape[0],height,width,no_channels))
  for i in tqdm(range(0,poses.shape[0])):
    channelidx = 0
    if use_depth:
      f = os.path.join(image_path, 'depth', str(i+1) + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
        raise Exception('Could not read depth image %s' % f)
      
      x_train[i, :, :, channelidx] = img1[:,:,0]
      channelidx += 1
    
    if use_normals:
      f = os.path.join(image_path, 'normal',str(i+1) + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
          raise Exception('Could not read normal image %s' % f)
      
      x_train[i, :, :, channelidx:channelidx+3] = img1
      channelidx += 3

    if use_intensity:
      f = os.path.join(image_path, 'intensity', str(i+1)+ '.npy')
      try:
        img1 = np.load(f)
      except IOError:
        f = os.path.join(image_path,  'intensity', str(i+1)+ '.npz')
        img1 = np.load(f)
      x_train[i, :, :, channelidx] = img1[:,:,0]
      channelidx += 1

  checkpoint_path = weights_filename_cp

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  

  x_train_,x_val_, y_train_, y_val_ = train_test_split(x_train, poses, test_size=val_rate, random_state=22, shuffle=shuffle_data)


  weights_filename = weights_filename_cp
  if len(weights_filename) > 0:
    if os.path.exists(weights_filename):
      model.load_weights(weights_filename)
      logger.info("     Beginning to fine tune for "+weights_filename)
  model.fit(x_train_, y_train_,validation_data=(x_val_,y_val_),epochs=no_epochs,batch_size=batch_size,callbacks=[cp_callback])

  model.save("./final.hdf5")

          
