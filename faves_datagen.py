import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import sys
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
	def __init__(self,data_dir,batch_size=32,shuffle=True, height=224, width=224,index_list=None):

		self.high_dir = os.path.join(data_dir, 'high')
		self.low_dir = os.path.join(data_dir, 'low')
		# print(self.low_dir)
		self.image_list = [os.path.join(self.high_dir,img) for img in os.listdir(self.high_dir)]
		self.labels_list=len(os.listdir(self.high_dir))*[[0,1]]
		image_list2=[os.path.join(self.low_dir,img) for img in os.listdir(self.low_dir)]
		self.image_list.extend(image_list2)
		self.labels_list.extend(len(os.listdir(self.low_dir))*[[1,0]])
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.indexes=index_list
		self.on_epoch_end()
		self.height = height
		self.width = width

	def __len__(self):
		return int(np.floor(len(self.indexes)/self.batch_size))

	def on_epoch_end(self):
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __getitem__(self,idx):

		indexes=self.indexes[self.batch_size*idx:self.batch_size*(idx+1)]
		image_list=[self.image_list[k] for k in indexes]
		labels_list=[self.labels_list[k] for k in indexes]
		images=np.empty((self.batch_size, self.height, self.width, 3))
		labels=np.empty((self.batch_size, 2))
		for i in range(len(image_list)):
			img_name = image_list[i]
			labels[i,]=np.array(labels_list[i])
			images[i,] = cv2.imread(img_name)/255.0
			
		return images,labels



def getGenerators(data_dir,train_batch_size,valid_batch_size,test_batch_size):

	image_list = os.listdir(os.path.join(data_dir,'low'))
	image_list.extend(os.listdir(os.path.join(data_dir,'high')))
	indexes = np.arange(len(image_list))
	train_indexes,test_indexes=train_test_split(indexes,train_size=4800)
	validation_indexes,test_indexes=train_test_split(test_indexes,train_size=0.5)
	
	train_generator=DataGenerator(data_dir,train_batch_size,index_list=train_indexes)
	valid_generator=DataGenerator(data_dir,valid_batch_size,index_list=validation_indexes)
	test_generator=DataGenerator(data_dir,test_batch_size,index_list=test_indexes)

	return train_generator,valid_generator,test_generator 
