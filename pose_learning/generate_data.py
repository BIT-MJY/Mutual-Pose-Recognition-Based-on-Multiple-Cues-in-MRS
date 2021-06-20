#!/usr/bin/env python3
# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: A generator which generates batches for keras training
import os
import random
import numpy as np

# from keras.utils import Sequence
from tensorflow.keras.utils import Sequence


class ImagePoseMapping(Sequence):
  """ This class is responsible for loading training/validation data. It
      can be used as keras generator object in e.g. model.fit_generator.
  """
  
  def __init__(self, image_path, imgfilenames1, poses,
                batch_size, height, width, no_channels=4,
               use_depth=True, use_normals=True, use_semantic=False,
               use_intensity=False):

    self.image_path = image_path
    self.batch_size = batch_size  # ****
    self.imgfilenames1 = imgfilenames1   # ****
    self.poses = poses
    self.n = poses.shape[0]  # ****
    self.height = height  # ****
    self.width = width  # ****
    self.no_channels = no_channels
    self.use_depth = use_depth
    self.use_normals = use_normals
    self.use_semantic = use_semantic
    self.use_intensity = use_intensity


  
  def __len__(self):
    """ Returns number of batches in the sequence. (overwritten method)
    """
    return int(np.ceil(self.n / float(self.batch_size)))
  
  def __getitem__(self, idx):
    """ Get a batch. (overwritten method)
    """
    
    maxidx = (idx + 1) * self.batch_size
    size = self.batch_size
    if maxidx > self.n:
      maxidx = self.n
      size = maxidx - idx * self.batch_size
    batch_f1 = self.imgfilenames1[idx * self.batch_size: maxidx]
    x1 = np.zeros((size, self.height, self.width, self.no_channels))

    
    for i in range(0, size):
      self.prepareOneInput(x1, i, batch_f1)
    
    
    y1 = self.poses[idx * self.batch_size: maxidx]

    return ([x1], y1)

  def prepareOneInput(self, x1, i, batch_f1):
    
    channelidx = 0
    if self.use_depth:
      f = os.path.join(self.image_path, 'depth', batch_f1[i] + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
        raise Exception('Could not read depth image %s' % f)
      
      x1[i, :, :, channelidx] = img1.reshape(self.height, self.width)
      channelidx += 1
    
    if self.use_normals:
      # normal map == channel 1..3
      f = os.path.join(self.image_path, 'normal', batch_f1[i] + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
          raise Exception('Could not read normal image %s' % f)
        
      x1[i, :, :, channelidx:channelidx+3] = img1
      channelidx += 3
    
    if self.use_semantic:
      # normal map == channel 1..3
      f = os.path.join(self.image_path, 'semantic', batch_f1[i] + '.npy')
      try:
        img1 = np.load(f)
      except IOError:  
          raise Exception('Could not read semantic image %s' % f)
        
      x1[i, :, :, channelidx] = img1
      channelidx += 1

    if self.use_intensity:
      f = os.path.join(self.image_path, 'intensity', batch_f1[i] + '.npy')
      try:
        img1 = np.load(f)
      except IOError:
        # Try to read format .npz
        f = os.path.join(self.image_path,  'intensity', batch_f1[i] + '.npz')
        img1 = np.load(f)
      img1 = img1/255
      x1[i, :, :, channelidx] = img1.reshape(self.height, self.width)
      channelidx += 1

