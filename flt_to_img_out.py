import tensorflow as tf
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

a_file = "/data/CT_images/train/images/00001501_img.flt"
raw_x = tf.io.read_file(a_file)
flt_x = tf.io.decode_raw(raw_x, tf.float32)
flt_x = tf.reshape(flt_x, [512,512,1])

plt.figure(figsize=(1,1))
plt.subplot(1,1,1)
plt.imshow(flt_x.numpy(), cmap='gray', vmin=0, vmax=1)
plt.title("heatmap of flt")
plt.axis("off")
plt.show()
