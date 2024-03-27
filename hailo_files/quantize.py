import torch.nn.functional as F
from tensorflow.python.eager.context import eager_mode
from multiprocessing import Process
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from hailo_sdk_client.runner.client_runner import ClientRunner
from hailo_sdk_common.targets.inference_targets import ParamsKinds



model_name = 'rpn_mobilenetv2_sim'
hailo_model_har_name = '{}.har'.format(model_name)
assert os.path.isfile(hailo_model_har_name), 'Please provide valid path for HAR file'

runner = ClientRunner(hw_arch='hailo8', har=hailo_model_har_name)







# #create postprocess and data for input size 255
# def preproc(image, output_height=255, output_width=255, resize_side=255):
#     ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
#     with eager_mode():
#         h, w = image.shape[0], image.shape[1]
#         scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
#         resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [255, 255])#, align_corners=False)
#         #cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
#         return tf.squeeze(resized_image)

# images_path = 'images/255/'
# images_list = [img_name for img_name in os.listdir(images_path) if
#                os.path.splitext(img_name)[1] == '.jpg']
# calib_dataset_255 = np.zeros((len(images_list), 255, 255, 3), dtype=np.uint8)
# for idx, img_name in enumerate(images_list):
#     img = np.array(Image.open(os.path.join(images_path, img_name)))
#     img_preproc = preproc(img)
#     calib_dataset_255[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
#     #print('img_name', img_name, 'idx = 

calib_dataset_255 = np.load('calib_dataset_255.npy')

# #create postprocess and data for input size 127
# def preproc(image, output_height=127, output_width=127, resize_side=127):
#     ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
#     with eager_mode():
#         h, w = image.shape[0], image.shape[1]
#         scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
#         resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [127, 127])#, align_corners=False)
#         #cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
#         return tf.squeeze(resized_image)

# images_path = 'images/127/'

# #images_path = 'own_images'

# images_list = [img_name for img_name in os.listdir(images_path) if
#                os.path.splitext(img_name)[1] == '.jpg']
# calib_dataset_127 = np.zeros((len(images_list), 127, 127, 3), dtype=np.uint8)
# for idx, img_name in enumerate(images_list):
#     img = np.array(Image.open(os.path.join(images_path, img_name)))
#     img_preproc = preproc(img)
#     calib_dataset_127[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
#     #print('img_name', img_name, 'idx = ',idx)

calib_dataset_127 = np.load('calib_dataset_127.npy')


csv_path = 'translated_params_hailo_centernet.csv'
quantized_model_har_path = '{}_quantized_model.har'.format(model_name)

alls_lines = [
    'model_optimization_config(calibration, batch_size=1, calibset_size=64)\n',  # Batch size is 8 by default
    'post_quantization_optimization(finetune, policy=disabled)\n'
    'post_quantization_optimization(bias_correction, policy=enabled)\n',
]


open('simple_script.alls','w').writelines(alls_lines)

runner.load_model_script('simple_script.alls')

hn_layers = runner.get_hn_dict()['layers']
calib_dataset_dict = {'rpn_mobilenetv2_sim/input_layer1': (calib_dataset_127), 'rpn_mobilenetv2_sim/input_layer2': (calib_dataset_255)} # In our case there is only one input layer
runner.optimize(calib_dataset_dict)

runner.save_har(quantized_model_har_path)
