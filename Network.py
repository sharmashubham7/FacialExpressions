import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, model_from_json



train_smile =  'train_smile2.csv'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

BATCH_SIZE = 4

train_image_dir = 'training_images'

file_path = []
for i in image_list:
               
    file_path.append(train_image_dir+'\\'+i)

label_list = list(df.iloc[:,1])


image_path_data = tf.data.Dataset.from_tensor_slices(np.array(file_path).reshape(-1,1))
label_data = tf.data.Dataset.from_tensor_slices(np.array(label_list).reshape(-1,1).astype('float32'))
total_input = tf.data.Dataset.zip((image_path_data, label_data)).batch(BATCH_SIZE).shuffle(1000)



def autoencoder():
    latent_dim = 512
    
    image_input = keras.Input(shape = (128,128,3), name = 'image_input')
    func_input = keras.Input(shape = (1,1), name = 'AU_input')
    
    x = layers.Conv2D(filters=32, kernel_size=5, strides=(2,2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(filters=64, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(filters=256, kernel_size=5, strides=(2,2), padding='same')(x)
    
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    
    img_encoded = layers.Dense(2*latent_dim, name = 'encoded_image')(x)
    flattened_au = layers.Flatten()(func_input)
    combined_input = layers.concatenate([img_encoded, flattened_au])


    x = layers.Dense(4096)(combined_input)
    x = layers.Dense(4096)(x)
    x = layers.Dense(16384)(x)
    x = layers.Reshape((8,8,256))(x)
    x = layers.Conv2DTranspose(filters=256, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    output_img= layers.Conv2DTranspose(filters = 3, kernel_size =5, strides= (1,1), padding = 'same', name = 'output_image')(x)
    
    x = layers.Conv2D(filters=32, kernel_size=5, strides=(2,2), padding='same')(output_img)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(filters=64, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    smile = layers.Dense(1, activation = 'softmax' , name= 'smile_identifier')(x)
    
    model_AE = keras.Model(inputs = [image_input, func_input], outputs = [output_img, smile])



    return model_AE



autoE = autoencoder()




AE_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def ae_loss(input_image, output_image, label, predicted):
    image_loss = tf.keras.losses.MSE(input_image, output_image)
    label_loss = cross_entropy(label, predicted)
    total_loss = (0.5*image_loss + 0.5*label_loss)
    return total_loss




def train_step(images, label):
    with tf.GradientTape() as AE_tape:
        output_image, predicted = autoE([images, label])
        loss = ae_loss(images, output_image, label, predicted)
        gradientsofAE = AE_tape.gradient(loss, autoE.trainable_variables)
        AE_optimizer.apply_gradients(zip(gradientsofAE, autoE.trainable_variables))




epochs = 10




def image_loader(paths):
    loaded_image = []
    for path in paths:
        path = np.array(path).astype(str)
        image = cv2.imread(str(path))/255.0
        loaded_image.append(image)
    return np.array(loaded_image)





def train(total_input):
    for epoch in range(epochs):
            autoE.save_weights('autoE_weights',overwrite = True)
            
            print('starting_epoch:', epoch)
            
            for count, batch_input in enumerate(total_input):
                
                if(count%1000==0):
                    print('.',end='')
                image_path_data, label = batch_input
                images = tf.map_fn(image_loader, image_path_data, dtype = tf.float32)
                images = np.array(images).reshape(-1,128,128,3)
                label  = np.array(label).reshape(-1,1).astype('float32')
                train_step(images, label)
            autoE.save_weights('autoE_weights',overwrite = True)
                
train(total_input)