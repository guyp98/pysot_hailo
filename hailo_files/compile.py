import torch.nn.functional as F
from tensorflow.python.eager.context import eager_mode
from multiprocessing import Process
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
import onnxruntime
from hailo_sdk_client.runner.client_runner import ClientRunner
from hailo_sdk_common.targets.inference_targets import ParamsKinds

model_name = 'rpn_mobilenetv2_sim'

quantized_model_har_path = f'{model_name}_quantized_model.har'

runner = ClientRunner(har=quantized_model_har_path)

hef = runner.compile()

file_name = f'{model_name}.hef'
with open(file_name, 'wb') as f:
    f.write(hef)