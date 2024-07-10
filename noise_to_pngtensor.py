import time
import tensorflow as tf 
import sys
import os
import glob #for unix filepath evaluation
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt #for image presentation

dataFilesGlob = '/data/CT_images/train/images/*.png'
dataSize = 5
dataSplit = 0.2
batchSize = 32

def process_png(filePath):
	data = tf.io.read_file(filePath)
	data = tf.image.decode_png(data, channels=1, dtype=tf.dtypes.uint8)
	data = tf.cast(data, tf.float32) / 255.
	return (data, filePath)

def poisson_noise_to_normalized_png(pngTensor, label):
	poissonTensor = tf.random.poisson(lam=0.075, shape=tf.shape(pngTensor), dtype=tf.float32)
	combinedTensor = pngTensor + poissonTensor
	combinedTensor = tf.clip_by_value(combinedTensor, 0.0, 1.0)
	return combinedTensor, label #preserves labels for perturbed data

def configure_for_performance(dataSet):
	dataSet = dataSet.map(lambda data, label: data) #isolate data from data,label tuple
	dataSet.cache()
	dataSet.shuffle(buffer_size=1000)
	dataSet.batch(batchSize)
	dataSet.prefetch(buffer_size=tf.data.AUTOTUNE)
	return dataSet

#get dataSize number of files
file_list_x = glob.glob(dataFilesGlob)
dataset_x = tf.data.Dataset.from_tensor_slices(file_list_x)

#train and validation split
valSize = tf.cast(dataSize * dataSplit, tf.dtypes.int64)
x_train = dataset_x.skip(valSize)
x_val = dataset_x.take(valSize)

#for label in x_train.take(1):
#	print("train:",label)
#for label in x_val.take(1):
#	print("val:",label)

#get normalized image data
x_train = x_train.map(process_png, num_parallel_calls=tf.data.AUTOTUNE)
x_train_pois = x_train.map(poisson_noise_to_normalized_png, num_parallel_calls=tf.data.AUTOTUNE)
x_val = x_val.map(process_png, num_parallel_calls=tf.data.AUTOTUNE)
x_val_pois = x_val.map(poisson_noise_to_normalized_png, num_parallel_calls=tf.data.AUTOTUNE)

#print(x_train.cardinality())
#for data, label in x_train.take(1):
#	print("train:",label)
#for data, label in x_val.take(1):
#	print("val:",label)


#establish foundations for speed!
x_train_PMD = configure_for_performance(x_train)
x_train_pois_PMD = configure_for_performance(x_train_pois)
x_val_PMD = configure_for_performance(x_val)
x_val_pois_PMD = configure_for_performance(x_val_pois)

xn = iter(x_val)
xn = next(xn)
xp = iter(x_val_pois)
xp = next(xp)

dataTensors = xn, xp
plt.figure(figsize=(2,1))
for i in (0,1):
	image, label = dataTensors[i]
	plt.subplot(2,1,i+1)
	plt.imshow(image.numpy(), cmap='gray', vmin=0, vmax=1)
	plt.title(label)
	plt.axis("off")
plt.show()
