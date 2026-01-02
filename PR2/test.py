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

import threading
import numpy as np
import ray
import os
import pynvml 


from Ray_ACNet import ACRDNet
import GroupLock


from testworker import Testworker
import scipy.signal as signal

import os
import pandas as pd
from Primal2Env import Primal2Env
from Primal2Observer import Primal2Observer
from Map_Generator import random_obstacle_generator
from parameters import *



RNN_SIZE=512


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
        
        fraction = 1
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



        path = "test3"#model_path
        checkpoint = tf.train.Checkpoint(model=global_network, world_optimizer=world_optimizer,policy_optimizer=policy_optimizer)

        checkpoint_manager=tf.train.CheckpointManager(checkpoint,model_path,1)

   
    if load_model == True:
        print ('Loading Model...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

        p=checkpoint_manager.latest_checkpoint
        print("checkpoint name:",p)


    MAP_SIZES = [30, 60]
    DENSITIES = [0.0, 0.15, 0.30]
    AGENT_COUNTS = [4, 8, 32, 64]
    NUM_EPISODES = 50
    MAX_STEPS = 256


    results = []
    for map_size in MAP_SIZES:
        for density in DENSITIES:
            for num_agents in AGENT_COUNTS:
                print(f"\n--- Testing: Map={map_size}, Density={density}, Agents={num_agents} ---")


                success_count = 0
                episode_lengths = []

                for ep in range(NUM_EPISODES):
                    # 環境の初期化 (Map_Generatorを使用)
                    env = Primal2Env(
                        num_agents=num_agents,
                        observer=Primal2Observer(OBS_SIZE, NUM_FUTURE_STEPS),
                        map_generator=random_obstacle_generator(
                            env_size=(map_size, map_size),
                            obstacle_density=(density, density, density)
                        ),
                        IsDiagonal=DIAG_MVMT,
                        isOneShot=IS_ONESHOT
                    )




                    workers = []
                    worker_threads = []
                    workerNames = ["worker_" + str(i+1) for i in range(NUM_THREADS)]#2+ int((NUM_THREADS-2)*max([min([(curriculum_level)/6.0, 1.0]), 0.0])))]
                    groupLock = GroupLock.GroupLock([workerNames, workerNames]) # TODO  


                    inference_lock = threading.Lock()       
                    coord = tf.train.Coordinator()
              
                    for a in range(num_agents):
                        agentID = a + 1

                        workers.append(Testworker(num_agents,agentID,
                                              env, global_network,
                                              groupLock,inference_lock))

                    for w in workers:
                        groupLock.acquire(0, w.name)
                        worker_work = lambda: w.work(coord)
                        t = threading.Thread(target=(worker_work))
                        t.start()
                        
                        worker_threads.append(t)

                    coord.join(worker_threads)

                    num_goals=0

                    for w in workers:
                        if w.isgoal:
                            num_goals+=1
                    is_success=False
                    if num_goals==num_agents:
                      success_count+=1
                      is_success=True

                    all_lengths = [w.length for w in workers]
                    max_length = max(all_lengths) if all_lengths else 256
                    episode_lengths.append(max_length)

                    print(f"Ep {ep+1}/{NUM_EPISODES}: Steps={max_length}, Success={is_success}")
                            
                avg_length = np.mean(episode_lengths)
                success_rate = (success_count / NUM_EPISODES) * 100
                
                results.append({
                    'MapSize': map_size,
                    'Density': density,
                    'Agents': num_agents,
                    'SuccessRate': success_rate,
                    'AvgLength': avg_length
                })

                df = pd.DataFrame(results)
                df.to_csv('test_results.csv', index=False)
                print("\none case completed. Results saved to test_results.csv")
                print(df)


    

if __name__ == "__main__":
    main()
