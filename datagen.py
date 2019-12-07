
import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from operations import *
class DataGenerator(Sequence):
	def __init__(self,data_dir,phone,phone_res,camera_res,batch_size=1,training=False,shuffle=True):

		self.lowQ_dir = os.path.join(data_dir, phone)
		self.highQ_dir = os.path.join(data_dir, 'canon')
		self.image_list = os.listdir(self.lowQ_dir)
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.lowQ_size=phone_res
		self.highQ_size=camera_res
		self.training=training
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.image_list)/self.batch_size))

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.image_list))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __getitem__(self,idx):
		indexes=self.indexes[self.batch_size*idx:self.batch_size*(idx+1)]
		image_list=[self.image_list[k] for k in indexes]
		lowQ_images=np.empty((self.batch_size,*self.lowQ_size,3),dtype=np.float32)
		highQ_images=np.empty((self.batch_size,*self.highQ_size,3),dtype=np.float32)
		highQ_images_coupled=np.empty((self.batch_size,*self.highQ_size,3),dtype=np.float32)
		for i in range(len(image_list)):
			img_name = image_list[i]
			lowQ_images[i,] =preprocess(cv2.imread(os.path.join(self.lowQ_dir, img_name)))
			highQ_images_coupled[i,] = preprocess(cv2.imread(os.path.join(self.highQ_dir, img_name)))
			img_name=np.random.choice(self.image_list)
			highQ_images[i,] = preprocess(cv2.imread(os.path.join(self.highQ_dir, img_name)))

		return lowQ_images, highQ_images,highQ_images_coupled

