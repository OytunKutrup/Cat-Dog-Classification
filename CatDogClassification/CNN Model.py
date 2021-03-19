import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time



#Verilerin konumunu ve kategorilerini olusturma
DATADIR = "C:/Users/gizli/Desktop/PetImages"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 50
training_data = []
#Verileri modeli eğitmek icin hazirlama
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        clas_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, clas_num])
            except Exception as e:
                pass

#Duzenledigimiz verilerin pickle kullanilarak kaydedilmesi
create_training_data()
pickle_out = open("training_data.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()

#Pickle kullanilarak duzenledigimiz verilerin yuklenmesi
pickle_in = open("training_data.pickle", "rb")
training_data = pickle.load(pickle_in)

#Verilerin karistirilmasi
random.shuffle(training_data)

#Verilere etiket atanmasi

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#Verilerin numpy array'e donusturulmesi
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Verileri ve etiketlerin pickle kullanilarak kaydedilmesi
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#Verilerin ve etiketlerin pickle kullanilarak yuklenmesi
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#Verinin normalizasyonu
X=np.array(X/255.0)
y=np.array(y)


#En iyi ogrenmeyi saglamak icin parametrelerin deneyerek bulunmasi
#Burda denemelerin sonucunda en iyi parametleri yazdim
dense_layers = [0] #0, 1, 2
layer_sizes = [64] #32, 64, 128
conv_layers = [2] #1, 2, 3
#CNN modelinin egitilmesi

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
#           Her bir denemeye isim atanmasi
            NAME = "{}-conv-{}-nodes{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#           Denemenin tensorboarda aktarilmasi
            tensorboard = TensorBoard(log_dir="C:\\Users\\gizli\\Desktop\\logs\\{}".format(NAME))
            model = Sequential()
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
#               Overfittingi engellemek icin dropout ekledim
                model.add(Dropout(0.2))

#           Output layer
            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
            model.fit(X,y, batch_size=32, epochs=7, validation_split=0.1, callbacks=[tensorboard])




#Olusturdugumuz en iyi sonuc veren modelin kaydetmek icin yaptım
model.save("2x64x0-CNN.model")