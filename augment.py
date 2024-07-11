import tensorflow as tf
import sys
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

#goals for augmentation: 4 rotation, 2 brightness adjustment, 1 noise addition
#create eight copies, rotate pairs, adjust brightness individually, stack, add noise to all of them

@tf.function
def poisson_noise_to_normalized_flt(fltTensor):
	poissonTensor = tf.random.poisson(lam=0.0025, shape=tf.shape(fltTensor), dtype=tf.float32)
	combinedTensor = fltTensor + poissonTensor
	combinedTensor = tf.clip_by_value(combinedTensor, 0.0, 1.0)
	return combinedTensor

@tf.function
def dim(fltTensor):
	dimmedTensor = tf.image.adjust_brightness(fltTensor, delta=-0.05)
	dimmedTensor = tf.clip_by_value(dimmedTensor, 0.0, 1.0)
	return dimmedTensor

a_file = "/data/CT_images/train/images/00001501_img.flt"
raw_x = tf.io.read_file(a_file)
flt_x = tf.io.decode_raw(raw_x, tf.float32)
flt_x = tf.reshape(flt_x, [512,512,1])

flt_r0_x = flt_x #would need to be a mapping if more than one image
flt_r1_x = tf.image.rot90(flt_x, k=1)
flt_r2_x = tf.image.rot90(flt_x, k=2)
flt_r3_x = tf.image.rot90(flt_x, k=3)

flt_rfull_x = tf.stack([flt_r0_x, flt_r1_x, flt_r2_x, flt_r3_x])
adjusted_brightness_x = tf.vectorized_map(dim, flt_rfull_x)
combined_x = tf.concat([flt_rfull_x, adjusted_brightness_x], axis=0)
augmented_x = tf.vectorized_map(poisson_noise_to_normalized_flt, combined_x)
dataset_augmented_x = tf.data.Dataset.from_tensor_slices(augmented_x)

plt.figure(figsize=(2,4))
i = 0
for flt in dataset_augmented_x:
	plt.subplot(2,4,i+1)
	plt.imshow(flt.numpy(), cmap='gray', vmin=0, vmax=1)
	plt.title(i)
	plt.axis("off")
	i += 1
plt.show()
