#!~/miniconda3/envs/neuralEnv/bin/python

import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import * # type: ignore
from tensorflow.keras.layers import * # type: ignore
from qkeras import *
from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm
import hls4ml
import matplotlib.pyplot as plt

from qkeras.utils import _add_supported_quantized_objects

import plotting

from hls4ml.model.profiling import numerical, get_ymodel_keras

# cargo variables de entorno
# Path donde se encuentra instalado Vitis HLS 
os.environ['PATH'] = '/tools/Xilinx/XilinxUnified_2022/Vitis_HLS/2022.2/bin:' + os.environ['PATH']

# Path para la maquina virtual
os.environ['PATH'] = '/tools/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls:' + os.environ['PATH']



# carga del modelo
co = {}
_add_supported_quantized_objects(co)

model = load_model('mnistPQKD.h5', custom_objects=co) # type: ignore



# ploteo distribución de weights
weights = np.concatenate([w.flatten() for w in model.get_weights()])

plt.figure(figsize=(10,2))
plt.hist(weights, bins=60, color='green', alpha=0.6)
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.title("Model MLP for MNIST - Weight Distribution")
plt.show()



# defino la configuración para HLS
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

print("-----------------------------------")
plotting.print_dict(hls_config)
print("-----------------------------------")



# configuro las distintas precisiones para cada capa
#   los nombres de las capas pueden ser distintas dependiendo del modelo cargado

# hls_config['LayerName']['fc1_input_input']['Precision'] = 'ap_fixed<16, 6>'  
# hls_config['LayerName']['fc1_input']['Precision'] = 'ap_fixed<8, 2>'  
# hls_config['LayerName']['relu_input']['Precision'] = 'ap_fixed<8, 3>'
# hls_config['LayerName']['softmax']['Strategy'] = 'Stable'

hls_config['Model']['Precision'] = 'ap_fixed<8,4>'



# creo la configuracion para hls4ml usando lo configurado anteriormente
cfg = hls4ml.converters.create_config(backend='Vitis')

# cfg['IOType']     = 'io_stream'   # Must set this if using CNNs!
cfg['HLSConfig']  = hls_config      # HLS configuraiton
cfg['KerasModel'] = model           # Keras model to be converted
cfg['OutputDir']  = 'hlsPrj/'       # Project name
cfg['Part'] = 'xc7z020clg484-1'     
# PYNQ-Z1 or Zedboard: xc7z020clg484-1  
# ARTIX-7 xc7a35tcsg325-1  
# MPSoC xczu4eg-sfvc784-2-e  xczu3eg-sfvc784-1-e

hls_model = hls4ml.converters.keras_to_hls(cfg)

hls_model.compile()



# construyo el modelo
hls_model.build(csim=False, export=False)
