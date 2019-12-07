
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from faves_datagen import *
import matplotlib.pyplot as plt

class Faves_model():

    def __init__(self,lr=0.00005):
        self.lr=lr
        self.class_list = ["Original","Tampered"]
        self.fc_layers = [1024, 1024]
        self.build_finetune_model(0.5,self.fc_layers,len(self.class_list))
        self.model_ready=None

    def build_finetune_model(self, dropout, fc_layers, num_classes):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        for fc in fc_layers:
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(dropout)(x)

        predictions = Dense(num_classes, activation='softmax')(x) 
        
        self.finetune_model = Model(inputs=base_model.input, outputs=predictions)
        adam = Adam(lr=self.lr)
        self.finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

    

    def train_model(self,epochs=100,data_dir='./drive/My Drive/flickr_dataset'):
    

        filepath="./checkpoints/" + "faves" + "_model_weights2.h5"
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")

        checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True)
        early_stopping=EarlyStopping(monitor='val_acc', min_delta=0.05, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        callbacks_list = [checkpoint,early_stopping]

        train_generator,validation_generator,test_generator=getGenerators(data_dir,25,2,2)
        history = self.finetune_model.fit_generator(train_generator, epochs=epochs, workers=8,
                                               shuffle=True, callbacks=callbacks_list,validation_data=validation_generator,validation_freq=5,use_multiprocessing=True)
        print("testing")
        print(self.finetune_model.evaluate_generator(test_generator))
        self.finetune_model.save_weights("./model_weights.h5")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./accuracy.jpg')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./loss.jpg')

        self.model_ready=True

    def load_finetune_model(self,model_path):
        self.finetune_model.load_weights(model_path)
        self.model_ready=True
    
    def predict(self,images):

        if self.model_ready :
            self.finetune_model.predict(images)
        else:
            raise Exception("train or load the model first")

    def faves_scorer(self,images):
        score=0.0
        if self.model_ready:
            predictions=self.finetune_model.predict(img,steps=1)[1]
            for p in predictions:
                score+=p[1]
            return score/float(len(images))
        else:
            raise Exception("train or load the model first")

    

if __name__=='__main__':
    obj=Faves_model()
    obj.train_model()
