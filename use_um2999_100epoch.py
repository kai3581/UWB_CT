import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shard3 import (
                    load_original_recon_dataset,
                    xy_dataset_from_xi_yi_datasets,
                    normalize_float32_tensor_0_1,
                    fltbatch_plt)
from custom_loss_functions import combined_loss

BATCH_SIZE = 50
MODEL = '/gscratch/uwb/bodhik/CT-CNN-Code/unet_model_0_2999_100epoch.h5'
IMAGE_DIR = '/gscratch/uwb/bodhik/CT-CNN-Code/tolocal'


X_DATA_FILE_PATTERN = (
    '/gscratch/uwb/CT_images/recons2024/60views/000037[0-4]*.flt')
Y_DATA_FILE_PATTERN = (
    '/gscratch/uwb/CT_images/recons2024/900views/000037[0-4]*.flt')
xi_dataset = load_original_recon_dataset(X_DATA_FILE_PATTERN)
yi_dataset = load_original_recon_dataset(Y_DATA_FILE_PATTERN)
abnormal_xy_dataset = xy_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset)
normalized_xy_dataset = (
    abnormal_xy_dataset.map(
        lambda x,y: (normalize_float32_tensor_0_1(x),
                     normalize_float32_tensor_0_1(y)),
                     num_parallel_calls=tf.data.AUTOTUNE))
tiled_normal_xy_dataset = normalized_xy_dataset.map(
            lambda x,y: (tf.tile(x, [1,1,3]),
                         tf.tile(y, [1,1,3])),
                         num_parallel_calls=tf.data.AUTOTUNE)

batched_dataset = tiled_normal_xy_dataset.batch(BATCH_SIZE)

loaded_model = tf.keras.models.load_model(
        MODEL, custom_objects={'combined_loss':combined_loss})

for x_batch, y_batch in batched_dataset.take(1):
    p_batch = loaded_model.predict_on_batch(x_batch)
    fltbatch_plt(x_batch, BATCH_SIZE, 'x', IMAGE_DIR)
    fltbatch_plt(p_batch, BATCH_SIZE, 'p', IMAGE_DIR)
    fltbatch_plt(y_batch, BATCH_SIZE, 'y', IMAGE_DIR)
