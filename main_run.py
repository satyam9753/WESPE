from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import datagen
import model
import cv2
import pickle
import time
import tqdm
from operations import *
train_data_folder='./dped/iphone/training_data'
# train_data_folder='./data/iphone'
phone='iphone'
test_data_folder='./dped/iphone/test_data/patches'
# test_data_folder='./data/iphone'
def main():

	phone_res=(100,100)
	camera_res=(100,100)
	num_images_per_step=100
	data_generator=datagen.DataGenerator(train_data_folder,phone,phone_res,camera_res,batch_size=30,training=True)
	main_model=model.WESPE(phone_res,camera_res)
	epochs=110
	test_generator=datagen.DataGenerator(test_data_folder,phone,phone_res,camera_res)
	train_data=[]
	test_data=[]
	for j in range(epochs):
		print('Epoch_'+str(j+1)+'/'+str(epochs))
		data_generator.on_epoch_end()
		data_list=tqdm.tqdm(range(min(num_images_per_step,len(data_generator))))		
		try:
			epoch_data=[]
			for i in data_list:
				x,y,y_coupled=data_generator[i]
				
				losses,y_generated=main_model.train_step(x,y)
				losses['avg_psnr'],losses['avg_ssim']=compare_imgs(postprocess(y_generated.numpy()),postprocess(y_coupled))
				epoch_data.append(losses)
				data_list.set_description(str(losses))
				data_list.refresh()
				time.sleep(0.01)
			train_data.append(epoch_data)

			if j%2==0:
				epoch_data={}
				for i in tqdm.tqdm(range(len(data_generator))):
					x,_,_=data_generator[i]
					for j in range(len(x)):
						cv2.imwrite('./drive/My Drive/dataset/dped/org'+str(i+j)+'.jpg',postprocess(x[j]))
					img=main_model.predict(x).numpy()
					for j in range(len(x)):
						cv2.imwrite('./drive/My Drive/dataset/dped/pred'+str(i+j)+'.jpg',postprocess(img[j]))
					epoch_data['avg_psnr'],epoch_data['avg_ssim']=compare_imgs(postprocess(img),postprocess(x))
				test_data.append(epoch_data)

			with open('data.pkl','wb') as file:
				pickle.dump((train_data,test_data),file)
		except Exception as e:
			print(str(e))

def compare_imgs(generated_imgs, target_imgs):
    s_sim=0.0
    p_snr=0.0

    for i in range(len(generated_imgs)):
    	s_sim+=ssim(generated_imgs[i],target_imgs[i],multichannel=True)
    	p_snr+=psnr(target_imgs[i],generated_imgs[i])

    return p_snr/len(generated_imgs), s_sim/len(generated_imgs)



if __name__=='__main__':
	main()
