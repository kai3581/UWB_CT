import os  #Importing needed libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shard3 import (
	get_basename,
	where_tensor_plt, 
	load_tfrecord_x_y_pair_recon_dataset,
	load_original_recon_dataset,
	xy_dataset_from_xi_yi_datasets,
        normalize_float32_tensor_0_1,###
        configure_for_performance,
        fltbatch_plt)

BATCH_SIZE = 50
BUFFER_SIZE = 500
EPOCH = 100
IMAGE_OUT_DIR = '/gscratch/uwb/bodhik/CT-CNN-Code/tolocal'
X_DATA_FILE_PATTERN = (
    '/gscratch/uwb/CT_images/RECONS2024/60views/0000[0-2]*.flt')
Y_DATA_FILE_PATTERN = (
    '/gscratch/uwb/CT_images/RECONS2024/900views/0000[0-2]*.flt')

        
#batchSize=50

#def configure_for_performance(dataset):
#	dataset = dataset.cache()
#	dataset = dataset.shuffle(buffer_size=500)
#	dataset = dataset.batch(batchSize)
#	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#	return dataset

#def fltbatch_plt(fltbatch, fltbatch_size, name):
#    for i in range(fltbatch_size):
#        fltname = f'./tolocal/{name}_{i}.png'
#        plt.figure(figsize=[1,1], dpi=600)
#        where_tensor_plt([1,1,1], fltbatch[i])
#        plt.savefig(fltname, dpi=600)
#        plt.close()



def ssim_loss(y_true, y_pred): #Defining a function of ssim loss image to image
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) #return the loss value

def mse_loss(y_true, y_pred): #Defining a mean squared error loss
    return tf.reduce_mean(tf.square(y_true - y_pred)) #Returning loss

def combined_loss(y_true, y_pred, alpha = 0.2, beta = 0.8): #Define a mixed loss with proportions alpha and beta
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred) #Return the sum of the weighted losses =1

def unet_model(input_size=(512, 512, 3)): #Defining the model
        inputs = tf.keras.Input(input_size)
            # Downsample
        ...
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Initial convolutional layer
        c1 = layers.Dropout(0.1)(c1) #Drops 10% of neurons from layer 1
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #Second convolution
        p1 = layers.MaxPooling2D((2, 2))(c1) #Max pools 2x2 regions

        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #Second Same as first section but filters x 2
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #Third same but filters x 2
        c3 = layers.Dropout(0.1)(c3)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #Fourth same but filters x 2
        c4 = layers.Dropout(0.1)(c4)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) 
            #Fourth has no maxpool and filters x 2
        c5 = layers.Dropout(0.1)(c5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #Undoes previous convolve
        u6 = layers.concatenate([u6, c4], axis=3) #First skip connection
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #Filtering convolutional layer
        c6 = layers.Dropout(0.1)(c6) #Drop 10% neurons from layer 6
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #Second convolution and filter

        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #Same as first but half filters
        u7 = layers.concatenate([u7, c3], axis=3) #Second skip connection
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.1)(c7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #Same but half filters
        u8 = layers.concatenate([u8, c2], axis=3)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #Same but half filters
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c9)


        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9) #Final 3 filter 1x1 kernel with sigmoid activation
        model = models.Model(inputs=inputs, outputs=[outputs]) #Compresses the function into a single variable "model"

        return model #Returns that variable
"""
# Directories
clean_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views/0000[0-2]*.flt' #1500 900views
dirty_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views/0000[0-2]*.flt' #1500 60views
"""

clean_dir = Y_DATA_FILE_PATTERN
dirty_dir = X_DATA_FILE_PATTERN

clean_im_and_id = load_original_recon_dataset(clean_dir)

dirty_im_and_id = load_original_recon_dataset(dirty_dir)#tuple type for mixed data



abnormal_xy_dataset = xy_dataset_from_xi_yi_datasets(
                dirty_im_and_id, clean_im_and_id)#tuple type, which deprecates slicing

normal_xy_dataset = abnormal_xy_dataset.map(#for sensible tensor dimensions
                lambda x, y: (normalize_float32_tensor_0_1(x), normalize_float32_tensor_0_1(y)),
                num_parallel_calls=tf.data.AUTOTUNE)

xy_dataset = normal_xy_dataset.map(#expand channels to 3 for grayscale -> rgb
            lambda x,y: (tf.tile(x, [1,1,3]), tf.tile(y, [1,1,3])),
            num_parallel_calls=tf.data.AUTOTUNE)
"""
for xy in xy_dataset:
    tf.print(xy[0].shape)
#shapetest
"""

xy_dataset_PMD = configure_for_performance(
                    xy_dataset, BUFFER_SIZE, BATCH_SIZE)
"""
for xy_batch in xy_dataset_PMD.take(1):
    tf.print(xy_batch[0].shape)
    plot_xytuple((xy_batch[0][0],xy_batch[1][0]), 'xyt.png')
"""#shapetest2: plot_xytuple is needed
xy_train_PMD = xy_dataset_PMD.skip(1)
for batch in xy_train_PMD:
    tf.print('1')

xy_test_PMD = xy_dataset_PMD.take(1)
for batch in xy_test_PMD:
    tf.print('2')
    
#defines model as the unet model
model = unet_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy']) #Compiles the model with adam optimizer and our mixed loss

# Train model
model.fit(xy_train_PMD, epochs=EPOCH, validation_data=xy_test_PMD) #Trains the dirty images with clean images as target

# Save model
model.save('unet_model.h5') #Saves the model
# Creates images run through the model

#for x_batch, y_batch in xy_test_PMD:
#    predictions_batch = model.predict_on_batch(x_batch)
#    fltbatch_plt(x_batch[0:2], 2, 'x_batch')
#    fltbatch_plt(predictions_batch[0:2], 2, 'p_batch')
#    fltbatch_plt(y_batch[0:2], 2, 'y_batch')

for x_batch, y_batch in xy_test_PMD:
    p_batch = model.predict_on_batch(x_batch)
    fltbatch_plt(x_batch[0:2], 2, 'x', IMAGE_OUT_DIR)
    fltbatch_plt(p_batch[0:2], 2, 'p', IMAGE_OUT_DIR)
    fltbatch_plt(y_batch[0:2], 2, 'y', IMAGE_OUT_DIR)
