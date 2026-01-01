import numpy as np
import tensorflow as tf
import os
import ray
import pickle
import collections

import pynvml

from Ray_ACNet import ACRDNet
from Runner import imitationRunner, RLRunner

import parameters
from parameters import *
import random




ray.init(num_gpus=1)


#tf.reset_default_graph()
print("Hello World")

'''
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction = 1.0 / (NUM_META_AGENTS - NUM_IL_META_AGENTS + 1)
config.gpu_options.allow_growth=True
'''


try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total / (1024 ** 2)  # MB単位
    pynvml.nvmlShutdown()
except pynvml.NVMLError_LibraryNotFound:
    print("NO GPU")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        
        fraction = 2.0 / (NUM_META_AGENTS - NUM_IL_META_AGENTS + 2)
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=fraction * tf.config.experimental.get_device_details(gpu)['memory_size'])]
                #get_device_detailsの返り値はGPUによるらしい、、、

                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=fraction * total_memory)]
            )
        
        
        #for gpu in gpus:
        #  tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)





def main():    
    with tf.device("/GPU:0"):
        world_optimizer = tf.keras.optimizers.Nadam(learning_rate=float(1))
        policy_optimizer= tf.keras.optimizers.Nadam(learning_rate=float(1))
        global_network = ACRDNet()

        #ダミーデータでのネットワーク構築
        dummy_obs=tf.zeros([1,1,4,11,11])
        dummy_goals=tf.zeros([1,1,3]) 
        dummy_latents=tf.zeros([1,1,RNN_SIZE])
        dummy_tentatives=tf.zeros([1,1,horizon-1,a_size])
        dummy_message=tf.zeros([1,1,1,RNN_SIZE+(horizon-1)*a_size])
        dummy_messages=tf.zeros([1,1,NUM_THREADS,RNN_SIZE+(horizon-1)*a_size])
        dummy_masks=tf.ones([1,1,1,NUM_THREADS],dtype=tf.bool)
        dummy_actions=tf.constant([[[1.0, 0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)

        global_network.encode(dummy_obs)
       
        global_network.communication.get_concrete_function(
            tf.TensorSpec(shape=[None, None, 1, RNN_SIZE+(horizon-1)*a_size], dtype=tf.float32),      # own_encoded_obs
            tf.TensorSpec(shape=[None, None, None, RNN_SIZE+(horizon-1)*a_size], dtype=tf.float32),   # all_messages (3次元目をNoneに！)
            tf.TensorSpec(shape=[None, None, 1, None], dtype=tf.bool)                 # mask (4次元目をNoneに！)
        )
        global_network.dynamics(dummy_latents,dummy_actions) 
        global_network.policy(dummy_latents)
        global_network.reward(dummy_latents,dummy_actions)
        global_network.q1(dummy_latents,dummy_actions)
        global_network.q2(dummy_latents,dummy_actions)




        checkpoint = tf.train.Checkpoint(model=global_network, world_optimizer=world_optimizer,policy_optimizer=policy_optimizer)

        checkpoint_manager=tf.train.CheckpointManager(checkpoint,model_path,1)

   
    if load_model == True:
        print ('Loading Model...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

        p=checkpoint_manager.latest_checkpoint
        print("checkpoint name:",p)