
import numpy as np
import os
import sys
import cv2
import re
import statistics
import pickle
from sklearn.model_selection import train_test_split
DATADIR = "./flickr"
OUT_DIR="./flickr_dataset"

def create_patches(images,no_of_patches=5):
    patches=np.empty((len(images)*5,224,224,3))
    for i in range(len(images)):
        img=images[i]
        shape_x,shape_y,_=img.shape
        patches_x=np.random.choice(range(0,shape_x-224),size=5)
        patches_y=np.random.choice(range(0,shape_y-224),size=5)
        for j in range(5):
            patches[i+j,]=resized[patches_x[j]:patches_x[j]+224,patches_y[j]:patches_y[j]+224,:]

    return patches


def dataset_creation():

    faves_score =[]
    image=[]
    size=len(os.listdir(DATADIR))
    for file_name in os.listdir(DATADIR):
            
        faves = re.findall(r"_0.(\d+).jpg", file_name)
        if not faves: 
          continue
        faves_score.append(float('0.'+faves[0]))                
    
    median = statistics.median(faves_score)
    print (median)
    img_name_list=os.listdir(DATADIR)
    for i in range(len(faves_score)):
        
        img_name=img_name_list[i]
        img_path = os.path.join(DATADIR , img_name)
        img=cv2.imread(img_path)
        resized=cv2.resize(img, (720,1280))
        patch_x = np.random.choice(range(0,1056)) 
        patch_y=np.random.choice(range(0,496))
        patches=resized[patch_x:patch_x+224,patch_y:patch_y+224,:]

        if (faves_score[i] > median):
            out_path=os.path.join(OUT_DIR,'high')
        else:
            out_path=os.path.join(OUT_DIR,'low')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path,str(i)+'.jpg'),patches)
    

if __name__=='__main__':

    dataset_creation()
    # train_patches,test_patches,train_labels,test_labels=train_test_split(patches,labels,train_size=4500)
    # test_patches,validation_patches,test_labels,validation_labels=train_test_split(test_patches,test_labels,train_size=0.5)

    # with open('validation_faves_dataset.pkl','wb') as file:
    #     pickle.dump((validation_patches,validation_labels),file)


    # with open('test_faves_dataset.pkl','wb') as file:
    #     pickle.dump((test_patches,test_labels),file)


    # with open('train_faves_dataset.pkl','wb') as file:
    #     pickle.dump((train_patches,train_labels),file)

