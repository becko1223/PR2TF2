import numpy as np
import tensorflow as tf
import os
import ray

import pynvml

from Ray_ACNet import ACNet
from Runner import imitationRunner, RLRunner

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
        fraction = 1.0 / (NUM_META_AGENTS - NUM_IL_META_AGENTS + 1)
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=fraction * tf.config.experimental.get_device_details(gpu)['memory_size'])]
                #get_device_detailsの返り値はGPUによるらしい、、、

                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=fraction * total_memory)]
            )
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)




# Create directories
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


global_step = 0
        
if ADAPT_LR:
    # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
    # we need the +1 so that lr at step 0 is defined
    lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
else:
    lr = tf.constant(LR_Q)



def apply_gradients(global_network, gradients, optimizer, curr_episode):
    optimizer.apply_gradients(zip(gradients,global_network.trainable_variables))
    if ADAPT_LR:
        lr = LR_Q / tf.sqrt(ADAPT_COEFF * curr_episode + 1.0)
        optimizer.learning_rate.assign(float(lr))
    else:
        optimizer.learning_rate.assign(LR_Q)
    global_step+=1

def writeImitationDataToTensorboard(global_summary, metrics, curr_episode):  
    with global_summary.as_default():
        tf.summary.scalar('Losses/Imitation loss',metrics[0],curr_episode)
        global_summary.flush()


def writeEpisodeRatio(global_summary, numIL, numRL,  curr_episode):
    with global_summary.as_default():
      
        current_learning_rate = LR_Q / tf.sqrt(ADAPT_COEFF * curr_episode + 1.0)

    if (numRL + numIL) != 0:
        RL_IL_Ratio = numRL / (numRL + numIL)
    else:
        RL_IL_Ratio = 0 
        tf.summary.scalar('Perf/Num IL Ep.', numIL,curr_episode)
        tf.summary.scalar('Perf/Num RL Ep.', numRL,curr_episode)
        tf.summary.scalar('Perf/ RL IL ratio Ep.', RL_IL_Ratio,curr_episode)
        tf.summary.scalar('Perf/Learning Rate', current_learning_rate,curr_episode)
        
        global_summary.flush()

    

def writeToTensorBoard(global_summary, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric
    
    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.mean(tensorboardData, axis=0))

        valueLoss, policyLoss, validLoss, entropyLoss, gradNorm, varNorm,\
            mean_length, mean_value, mean_invalid, \
            mean_stop, mean_reward, mean_finishes = tensorboardData
        
    else:
        firstEpisode = tensorboardData[0]
        valueLoss, policyLoss, validLoss, entropyLoss, gradNorm, varNorm, \
            mean_length, mean_value, mean_invalid, \
            mean_stop, mean_reward, mean_finishes = firstEpisode

        
    summary = tf.Summary()

    with global_summary.as_default():
        tf.summary.scalar('Perf/Reward',mean_reward,curr_episode)
        tf.summary.scalar('Perf/Targets Done',mean_finishes,curr_episode)
        tf.summary.scalar('Perf/Length',mean_length,curr_episode)
        tf.summary.scalar('Perf/Valid Rate',(mean_length-mean_invalid)/mean_length,curr_episode)
        tf.summary.scalar('Perf/Stop Rate',mean_stop/mean_length,curr_episode)

        tf.summary.scalar('Losses/Value Loss',valueLoss,curr_episode)
        tf.summary.scalar('Losses/Policy Loss',policyLoss,curr_episode)
        tf.summary.scalar('Losses/Valid Loss',validLoss,curr_episode)
        tf.summary.scalar('Losses/Entropy Loss',entropyLoss,curr_episode)
        tf.summary.scalar('Losses/Grad Norm',gradNorm,curr_episode)
        tf.summary.scalar('Losses/Var Norm',varNorm,curr_episode)
        global_summary.flush()


    
def main():    
    with tf.device("/GPU:0"):
        optimizer = tf.keras.optimizers.Nadam(learning_rate=float(lr))
        global_network = ACNet()
        #global_network.build([(None,11,11,11),(None,2),(2,1,512)])
        dummy_input=tf.zeros((1,11,11,11))
        dummy_goalpos=tf.zeros((1,3))
        dummy_state=[global_network.h0,global_network.c0]
        global_network(dummy_input,dummy_goalpos,dummy_state)

        global_summary = tf.summary.create_file_writer(train_path)
        checkpoint = tf.train.Checkpoint(model=global_network, optimizer=optimizer)
        checkpoint_manager=tf.train.CheckpointManager(checkpoint,model_path,1)

   
    if load_model == True:
        print ('Loading Model...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

        p=checkpoint_manager.latest_checkpoint
        p=p[p.find('-')+1:]
        p=p[:p.find('.')]
        curr_episode=int(p)

        print("curr_episode set to ",curr_episode)
    else:
        curr_episode = 0


        
        # launch all of the threads:
    
        il_agents = [imitationRunner.remote(i) for i in range(NUM_IL_META_AGENTS)]
        rl_agents = [RLRunner.remote(i) for i in range(NUM_IL_META_AGENTS, NUM_META_AGENTS)]
        meta_agents = il_agents + rl_agents

        

        

        weights=global_network.get_weights()


        
        # launch the first job (e.g. getGradient) on each runner
        jobList = [] # Ray ObjectIDs 
        for i, meta_agent in enumerate(meta_agents):
            jobList.append(meta_agent.job.remote(weights, curr_episode))
            curr_episode += 1

        tensorboardData = []


        IDs = [None] * NUM_META_AGENTS

        numImitationEpisodes = 0
        numRLEpisodes = 0
        try:
            while True:
                # wait for any job to be completed - unblock as soon as the earliest arrives
                done_id, jobList = ray.wait(jobList)
                
                # get the results of the task from the object store
                #jobResults, metrics, info = ray.get(done_id)[0]


                res = ray.get(done_id)[0]

                if not res["ok"]:
                    # 例外が発生しているので中身を表示して中断
                    print("Remote job error:")
                    print(f"Type: {res['error_type']}")
                    print(res["traceback"])
                    break  # or continue, or raise 例外を再発生させるなど対応

                # 成功時は result の中身をアンパック
                jobResults, metrics, info = res["result"]







                # imitation episodes write different data to tensorboard
                if info['is_imitation']:
                    if jobResults:
                        writeImitationDataToTensorboard(global_summary, metrics, curr_episode)
                        numImitationEpisodes += 1
                else:
                    if jobResults:
                        tensorboardData.append(metrics)
                        numRLEpisodes += 1


                # Write ratio of RL to IL episodes to tensorboard
                writeEpisodeRatio(global_summary, numImitationEpisodes, numRLEpisodes, curr_episode)

                
                if JOB_TYPE == JOB_OPTIONS.getGradient:
                    if jobResults:
                        for gradient in jobResults:
                            apply_gradients(global_network, gradient, optimizer, curr_episode)

                    
                elif JOB_TYPE == JOB_OPTIONS.getExperience:
                    print("not implemented")
                    assert(1==0)
                else:
                    print("not implemented")
                    assert(1==0)


                # Every `SUMMARY_WINDOW` RL episodes, write RL episodes to tensorboard
                if len(tensorboardData) >= SUMMARY_WINDOW:
                    writeToTensorBoard(global_summary, tensorboardData, curr_episode)
                    tensorboardData = []
                    
                # get the updated weights from the global network
                
                weights = global_network.get_weights()
                curr_episode += 1

                # start a new job on the recently completed agent with the updated weights
                jobList.extend([meta_agents[info['id']].job.remote(weights, curr_episode)])

                
                if curr_episode % 100 == 0:
                    print ('Saving Model', end='\n')
                    #checkpoint_numberのところにエピソードナンバーを保存しておく
                    checkpoint_manager.save(checkpoint_number=curr_episode)
                    print ('Saved Model', end='\n')

                
                    
        except KeyboardInterrupt:
            print("CTRL-C pressed. killing remote workers")
            for a in meta_agents:
                ray.kill(a)


if __name__ == "__main__": 
    main()
