import tensorflow as tf
import os
import sys
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

#goals for augmentation: 4 rotation, 2 brightness adjustment, 1 noise addition
#create eight copies, rotate pairs, adjust brightness individually, stack, add noise to all of them
#loading and storing: load a tensor, label pair to a tfrecord, must additionally be given a directory to load
	#save steps: glob the files to process, change the base readme to reflect this new set, change directory to where these
	#		files will be and create the readme with info on them,
	#		create a list_files dataset, read the data in with the map function
	#		package into tfrecord examples and write to files in the directory

	#load steps: get the name of the most recent dataset if left blank, error out if it doesn't exist, otherwise load the spec
	#		ified dataset into the filename dataset
	#		parse into tensor, label pairs,
	#		return them

tf.random.set_seed(1971)

@tf.function
def poisson_noise_to_normalized_flt(fltTensor):
	poissonTensor = tf.random.poisson(lam=0.0025, shape=tf.shape(fltTensor), dtype=tf.float32)
	combinedTensor = fltTensor + poissonTensor
	combinedTensor = tf.clip_by_value(combinedTensor, 0.0, 1.0)
	return combinedTensor

@tf.function
def random_uniform_brightness_shift(fltTensor):
	ur_delta = tf.random.uniform([], minval=-0.1, maxval=0.1, dtype=tf.float32)
	shiftedTensor = tf.image.adjust_brightness(fltTensor, delta=ur_delta)
	shiftedTensor = tf.clip_by_value(shiftedTensor, 0.0, 1.0)
	return shiftedTensor

a_file = "/data/CT_images/train/images/00001501_img.flt"

raw_x = tf.io.read_file(a_file)
flt_x = tf.io.decode_raw(raw_x, tf.float32)
flt_x = tf.reshape(flt_x, [512,512,1])
flt_label_os_x = os.path.basename(a_file)
flt_label_x = tf.constant(flt_label_os_x)

dataset_x = tf.data.Dataset.from_tensors((flt_x, flt_label_x))
for x, label_x in dataset_x:
	print(x.shape," ",label_x.shape)

#starting tfrecord practice

#for packing into a tfrecord format

#@tf.py_function(Tout=tf.string)
for flt_x, flt_label_x in dataset_x:
	serialized_flt_x = tf.io.serialize_tensor(flt_x)
	feature_of_float32_x = tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[serialized_flt_x.numpy()]))

	serialized_flt_label_x = tf.io.serialize_tensor(flt_label_x)
	feature_of_bytes_x = tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[serialized_flt_label_x.numpy()]))

	feature_x = {
		'data': feature_of_float32_x,
		'label': feature_of_bytes_x
	}

	example_proto_x = tf.train.Example(features = tf.train.Features(feature=feature_x))
	packaged_example_proto_x = example_proto_x.SerializeToString()

	ext = tf.constant(".tfrecord")
	filename = tf.strings.join([flt_label_x,ext])
	tf.io.TFRecordWriter(filename.numpy().decode('utf-8')).write(packaged_example_proto_x)

	
#unpacking the tfrecord file!

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames) #loads the tensor filenames

packaged_feature_x = {
	'data': tf.io.FixedLenFeature([], tf.string),
	'label': tf.io.FixedLenFeature([], tf.string)
}

for example_x in raw_dataset:
	parsed_example_x = tf.io.parse_single_example(example_x, packaged_feature_x)
	x = tf.io.parse_tensor(parsed_example_x['data'], out_type=tf.float32)
	x_label = tf.io.parse_tensor(parsed_example_x['label'], out_type=tf.string)

print(x_label.shape,' ',x.shape)
plt.figure(figsize=(1,1))
plt.subplot(1,1,1)
plt.imshow(x, cmap='gray', vmin=0, vmax=1)
plt.title(x_label.numpy().decode('utf-8'))
plt.axis("off")
plt.show()
