import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh,relu,softmax,sigmoid
from tensorflow.keras.layers import PReLU
import tensorflow.keras
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D,Conv2D,Flatten
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import Huber as huber_loss
from tensorflow.keras.losses import Reduction
import cv2

class ConvBlock(tensorflow.keras.layers.Layer):
	def __init__(self):
		super(ConvBlock,self).__init__()
		self.conv1 = Conv2D(64, (3,3),use_bias=True, padding='same')
		self.conv2 = Conv2D(64, (3,3),use_bias=True, padding='same')
		self.bn1 = tensorflow.keras.layers.BatchNormalization(axis=-1)
		self.bn2 = tensorflow.keras.layers.BatchNormalization(axis=-1)

	def call(self,inputs):
		y = relu(self.bn1(self.conv1(inputs)))
		y = relu(self.bn2(self.conv2(y)))+ inputs
		return y

class GaussianBlur(tensorflow.keras.layers.Layer):
	def __init__(self):
		super(GaussianBlur,self).__init__()

		kernel_size = 3  # set the filter size of Gaussian filter
		kernel_weights = np.asarray([[0.03797616, 0.044863533, 0.03797616],[0.044863533, 0.053, 0.044863533],[0.03797616, 0.044863533, 0.03797616]])

		in_channels = 3 
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		self.g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same',weights=[kernel_weights])
		self.g_layer.trainable = False 
		

	def call(self,inputs):
		g_layer_out = self.g_layer(inputs)  
		return g_layer_out


class GrayScale(tensorflow.keras.layers.Layer):
	def __init__(self):
		super(GrayScale,self).__init__()

	def call(self,image):
		# print(image.shape)
		r, g, b = image[:,:,:,0], image[:,:,:,1], image[:,:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		gray=tf.reshape(gray,(gray.shape[0],gray.shape[1],gray.shape[2],1))
		return gray

class Generator(tensorflow.keras.layers.Layer):

	def __init__(self,input_shape, scope="generator"):
		super(Generator, self).__init__()
		self.scope = scope
		with tf.name_scope(scope) as tf_scope:
			self.conv1 = Conv2D(64, (9,9),input_shape=(100,100) ,use_bias=True,padding='same')
			self.block1 = ConvBlock()
			self.block2=ConvBlock()
			self.block3=ConvBlock()
			self.block4=ConvBlock()
			self.conv2 = Conv2D(64, (3,3),use_bias=True, padding='same')
			self.conv3 = Conv2D(64, (3,3),use_bias=True, padding='same')
			self.conv4 = Conv2D(3, (9,9),use_bias=True, padding='same')
	def call(self,inputs):
		y=relu(self.conv1(inputs))
		y=self.block4(self.block3(self.block2(self.block1(y))))
		temp=relu(self.conv3(relu(self.conv2(y))))
		return tanh(self.conv4(temp))



class Discriminator(tensorflow.keras.layers.Layer):
	def __init__(self,input_shape, scope="discriminator"):
		super(Discriminator,self).__init__()
		with tf.name_scope(scope) as tf_scope:
			self.conv1=Conv2D(48, (11, 11),use_bias=True, input_shape=input_shape,strides=4, padding='same')
			self.relu1=PReLU()
			self.conv2=Conv2D(128, (5,5),use_bias=True, strides=2, padding='same')
			self.bn1=tensorflow.keras.layers.BatchNormalization(axis=-1)
			self.relu2=PReLU()
			self.conv3=Conv2D(192, (3,3),use_bias=True, strides=1, padding='same')
			self.bn2=tensorflow.keras.layers.BatchNormalization(axis=-1)
			self.relu3=PReLU()
			self.conv4=Conv2D(192, (3,3),use_bias=True, strides=1, padding='same')
			self.bn3=tensorflow.keras.layers.BatchNormalization(axis=-1)
			self.relu4=PReLU()
			
			self.conv5=Conv2D(128, (3,3),use_bias=True, strides=2, padding='same')
			self.bn4=tensorflow.keras.layers.BatchNormalization(axis=-1)
			self.relu5=PReLU()
			self.flatten=Flatten()
			self.fc = tensorflow.keras.layers.Dense(1024)
			self.relu6=PReLU()
			self.out = tensorflow.keras.layers.Dense(2) 


	def call(self, inputs):
		y = self.relu1(self.conv1(inputs))
		y = self.relu2(self.bn1(self.conv2(y)))
		y = self.relu3(self.bn2(self.conv3(y)))
		y = self.relu4(self.bn3(self.conv4(y)))
		y = self.relu5(self.bn4(self.conv5(y)))
		y=self.relu6(self.fc(self.flatten(y)))
		return self.out(y)

def tvlossfn(images):
	return tf.reduce_sum(tf.image.total_variation(images))/30000

def softmax_loss(x,y):
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=y))
def content_losses(x,y):
	return tf.reduce_mean(tf.square(x - y))/30000

class WESPE:
	def __init__(self,gen_input_shape,disc_input_shape):
		self.generator_g=Generator(gen_input_shape, scope="generator_g")
		self.generator_g.trainable=True
		
		self.generator_f=Generator(gen_input_shape, scope="generator_f")
		self.generator_f.trainable=True

		self.discriminator_c=Discriminator((100,100,3), scope="discriminator_c")
		self.discriminator_c.trainable=True

		self.discriminator_t=Discriminator((100,100,1), scope="discriminator_t")
		self.discriminator_t.trainable=True

		self.blur=GaussianBlur()
		self.blur.trainable=False
		# self.huber_obj=
		# self.content_loss=huber_loss()

		self.tv_loss=tvlossfn
		# self.texture_loss=tensorflow.keras.losses.categorical_crossentropy
		# self.color_loss=tensorflow.keras.losses.categorical_crossentropy
		self.content_loss=content_losses
		self.texture_loss=softmax_loss
		self.color_loss=softmax_loss


		self.gray=GrayScale()
		self.gray.trainable=False
		self.mobilenet=MobileNetV2(input_shape=(96,96,3),include_top=False)
		self.mobilenet.trainable=False
		self.gen_g_optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
		self.gen_f_optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
		self.disc_c_optimizer=tensorflow.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
		self.disc_t_optimizer=tensorflow.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)


	def train_step(self,x,y):

		x=tf.convert_to_tensor(x)
		y=tf.convert_to_tensor(y)
		batch_size=x.shape[0]
		with tf.GradientTape(persistent=True) as tape:
			
			self.generator_g.trainable=True
			self.generator_f.trainable=True

			y_fake=self.generator_g(x)

			x_fake=self.generator_f(y_fake)
			pos_indexes=tf.convert_to_tensor(np.asarray([[1.0,0.0]]*batch_size,dtype=np.float32))
			neg_indexes=tf.convert_to_tensor(np.asarray([[0.0,1.0]]*batch_size,dtype=np.float32))


			mobilenet_x_true=self.mobilenet(tf.image.resize(x,tf.constant([96,96])))
			mobilenet_x_fake=self.mobilenet(tf.image.resize(x_fake,tf.constant([96,96])))
			# print(y_fake.shape)
			y_real_blur=self.blur(y)
			y_fake_blur=self.blur(y_fake)

			y_fake_blur_pred=self.discriminator_c(y_fake_blur)
			y_real_blur_pred=self.discriminator_c(y_real_blur)

			y_fake_gray=self.gray(y_fake)
			y_real_gray=self.gray(y)
			y_fake_gray_pred=self.discriminator_t(y_fake_gray)
			y_real_gray_pred=self.discriminator_t(y_real_gray)

			content_loss=self.content_loss(mobilenet_x_fake,mobilenet_x_true)
			tv_loss=self.tv_loss(y_fake)
			dc_loss_g=self.color_loss(y_real_blur_pred,y_fake_blur_pred)
			dt_loss_g=self.texture_loss(y_real_blur_pred,y_fake_gray_pred)
			net_loss=content_loss+10*tv_loss+ 0.005*(dc_loss_g+dt_loss_g)

		grads = tape.gradient(net_loss, self.generator_g.trainable_weights)
		self.gen_g_optimizer.apply_gradients(zip(grads, self.generator_g.trainable_weights))
		grads = tape.gradient(net_loss, self.generator_f.trainable_weights)
		self.gen_f_optimizer.apply_gradients(zip(grads, self.generator_f.trainable_weights))
		


		with tf.GradientTape(persistent=True) as tape:
			self.discriminator_c.trainable=True
			self.discriminator_t.trainable=True


			self.generator_g.trainable=False
			self.generator_f.trainable=False

			y_fake_blur_pred=self.discriminator_c(y_fake_blur)
			y_real_blur_pred=self.discriminator_c(y_real_blur)
			dc_loss=self.color_loss(neg_indexes,y_fake_blur_pred)+self.color_loss(pos_indexes,y_real_blur_pred)
			y_fake_gray_pred=self.discriminator_t(y_fake_gray)
			y_real_gray_pred=self.discriminator_t(y_real_gray)
			dt_loss=self.texture_loss(neg_indexes,y_fake_gray_pred)+self.texture_loss(pos_indexes,y_real_gray_pred)

		grads=tape.gradient(dt_loss,self.discriminator_t.trainable_weights)
		self.disc_t_optimizer.apply_gradients(zip(grads,self.discriminator_t.trainable_weights))
		grads=tape.gradient(dc_loss,self.discriminator_c.trainable_weights)
		self.disc_c_optimizer.apply_gradients(zip(grads,self.discriminator_c.trainable_weights))

		return {'net_loss':net_loss.numpy(), 'dc_loss':dc_loss.numpy(),'dt_loss':dt_loss.numpy()},y_fake



	def predict(self,x):
		self.generator_f.trainable=False
		self.generator_g.trainable=False
		return self.generator_g(x)
