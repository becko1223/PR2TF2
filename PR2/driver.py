import numpy as np
import tensorflow as tf
import os
import ray
import pickle

import pynvml

from Ray_ACNet import ACRDNet
from Runner import imitationRunner, RLRunner

import parameters
from parameters import *
import random

GRAD_CLIP = 10.0


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




# Create directories
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


global_step = 0 #これはcurrent_episodeと同じ。混在していてちょっと良くない。

global_mean_finishes=0
global_10epi_finishes_0=0
global_10epi_finishes_1=0
        
if ADAPT_LR:
    # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
    # we need the +1 so that lr at step 0 is defined
    lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
else:
    lr = tf.constant(LR_Q)



def apply_gradients(global_network, gradients, world_optimizer,policy_optimizer, curr_episode):

    variables_for_actor=global_network.policy_dense1.trainable_variables+global_network.policy_dense2.trainable_variables+global_network.policy_dense3.trainable_variables
    actor_variable_names = set([v.name for v in variables_for_actor])
    all_trainable_variables = global_network.trainable_variables
    variables_except_for_actor = [
        v for v in all_trainable_variables 
        if v.name not in actor_variable_names
    ]
    
    if (isinstance(gradients,tuple)):
        world_optimizer.apply_gradients(zip(gradients[0],variables_except_for_actor))
        policy_optimizer.apply_gradients(zip(gradients[1],variables_for_actor))
    else:
        world_optimizer.apply_gradients(zip(gradients,global_network.trainable_variables))
    if ADAPT_LR:
        lr = LR_Q / tf.sqrt(ADAPT_COEFF * curr_episode + 1.0)
        world_optimizer.learning_rate.assign(float(lr))
        policy_optimizer.learning_rate.assign(float(lr))
    else:
        world_optimizer.learning_rate.assign(LR_Q)
        policy_optimizer.learning_rate.assign(LR_Q)
    global global_step
    global_step+=1


@tf.function
def tape_calc(global_network,batch_obs, batch_goals, batch_rewards, batch_actions, batch_states, batch_valids):

    variables_for_actor=global_network.policy_dense1.trainable_variables+global_network.policy_dense2.trainable_variables+global_network.policy_dense3.trainable_variables
    actor_variable_names = set([v.name for v in variables_for_actor])
    all_trainable_variables = global_network.trainable_variables
    variables_except_for_actor = [
        v for v in all_trainable_variables 
        if v.name not in actor_variable_names
    ]

    rhos=tf.convert_to_tensor([[rho**i for i in range(horizon)] for _ in range(batch_size)])
    

    #アクター以外訓練
    with tf.GradientTape() as tape:

        def dynamics(carry, elem):
            elem=tf.expand_dims(elem,axis=1)
            prev_latents,_=carry
            latents = global_network.dynamics(prev_latents,elem)
            rewards= global_network.reward(prev_latents,elem)
            return (latents,rewards)
        
        batch_actions_T = tf.transpose(batch_actions[:, :], [1, 0, 2])  # [horizon, batch, action_dim]

        latent_init,batch_states_step1=global_network.encode(batch_obs[:, 0:1],batch_goals[:,0:1],tf.reshape(batch_states[:,0],[-1,512]),tf.reshape(batch_states[:,1],[-1,512]))

        #latent,rewardの予測値
        batch_latent_preds,batch_reward_preds = tf.scan(  #[horizon,batch,1,dim]
            fn=dynamics,
            elems=batch_actions_T,     
            initializer=(latent_init,tf.zeros([batch_size,1,1], dtype=tf.float32))
            )
        
        batch_latent_preds = tf.squeeze(batch_latent_preds, axis=2)     #長さhorizon
        batch_latent_preds = tf.transpose(batch_latent_preds, [1, 0, 2])  # [h,b,dim]to[b,h,dim]

        batch_reward_preds = tf.squeeze(batch_reward_preds, axis=2)
        batch_reward_preds = tf.transpose(batch_reward_preds, [1,0,2])
        batch_reward_preds=tf.squeeze(batch_reward_preds) #[b,h,1] to [b,h,]
    
        #valueの予測値を出す
        batch_q1value_preds=global_network.q1(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],batch_actions[:,:])
        batch_q2value_preds=global_network.q2(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],batch_actions[:,:])
        batch_q1value_preds=tf.squeeze(batch_q1value_preds)
        batch_q2value_preds=tf.squeeze(batch_q2value_preds)


        #latentのターゲットを出す(b,s,h,w,c)
        batch_latent_targets,_=global_network.encode(batch_obs[:,1:],batch_goals[:,1:],batch_states_step1[0],batch_states_step1[1])


        #q target出す
        policy=global_network.policy(batch_latent_preds)
        policy=tf.clip_by_value(policy,-10.0,10.0)
        policy=tf.nn.softmax(policy)
        logits = tf.math.log(policy + 1e-10)
        B = tf.shape(logits)[0]
        H = tf.shape(logits)[1]
        A = tf.shape(logits)[2]
        flat_logits = tf.reshape(logits, [-1, A])
        flat_actions = tf.random.categorical(flat_logits, num_samples=1, dtype=tf.int64)
        next_actions = tf.reshape(flat_actions, [B, H, 1])

        next_actions = tf.squeeze(next_actions, axis=-1)
        next_actions=tf.one_hot(next_actions,a_size)

        q1_next=global_network.q1(batch_latent_preds,next_actions)
        q2_next=global_network.q2(batch_latent_preds,next_actions)

        q_next=tf.math.minimum(q1_next,q2_next)
        q_next=tf.squeeze(q_next)
        q_target=batch_rewards[:,:]+gammma_tdmpc*q_next

            

        reward_loss=tf.reduce_mean(rhos*tf.square(batch_reward_preds-batch_rewards[:,:]))
        q1value_loss=tf.reduce_mean(rhos*tf.square(q_target-batch_q1value_preds))
        q2value_loss=tf.reduce_mean(rhos*tf.square(q_target-batch_q2value_preds))
        
        consistency_loss=tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*tf.square(batch_latent_targets-batch_latent_preds))

        total_loss=0.5*reward_loss+0.1*(q1value_loss+q2value_loss)+2.0*consistency_loss
    world_grads=tape.gradient(total_loss,variables_except_for_actor)


    with tf.GradientTape() as tape:
        
        policy=global_network.policy(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1])
        policy=tf.clip_by_value(policy,-10.0,10.0)
        batch_policies_sig=tf.sigmoid(policy)
        policy=tf.nn.softmax(policy)
        #next_actions=tf.map_fn(lambda probs: tf.random.categorical(probs, 1),elems=policy,dtype=tf.int64)  

        logits = tf.math.log(policy + 1e-10)
        B = tf.shape(logits)[0]
        H = tf.shape(logits)[1]
        A = tf.shape(logits)[2]
        flat_logits = tf.reshape(logits, [-1, A])
        flat_actions = tf.random.categorical(flat_logits, num_samples=1, dtype=tf.int64)
        next_actions = tf.reshape(flat_actions, [B, H, 1])

        next_actions = tf.squeeze(next_actions, axis=-1)
        next_actions=tf.one_hot(next_actions,a_size)
        q1_next=global_network.q1(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],next_actions)
        q2_next=global_network.q2(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],next_actions)
        batch_q=tf.math.minimum(q1_next,q2_next)
        batch_q=tf.squeeze(batch_q) 
        
        


        policy_loss=-tf.reduce_mean(rhos*batch_q)
        
        valid_loss=-tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*(batch_valids[:,:]*tf.math.log(tf.clip_by_value(batch_policies_sig, 1e-10, 1.0))+(1-batch_valids[:,:])*tf.math.log(tf.clip_by_value(1-batch_policies_sig,1e-10,1.0))))
        entropy=-tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*policy * tf.math.log(tf.clip_by_value(policy, 1e-10, 1.0)))

        total_loss=0.5*policy_loss+16*valid_loss+entropy
    
    policy_grads=tape.gradient(total_loss,variables_for_actor)
    return world_grads,policy_grads,[reward_loss,(q1value_loss+q2value_loss)/2.0,consistency_loss,policy_loss,valid_loss,entropy]



def update(global_network, obs,goals,actions,rewards,states,valids, world_optimizer,policy_optimizer, curr_episode):

    batch_obs = tf.convert_to_tensor(obs,dtype=tf.float32) 
    batch_goals = tf.convert_to_tensor(goals,dtype=tf.float32)
    batch_rewards=tf.convert_to_tensor(rewards,dtype=tf.float32)
    batch_actions=tf.convert_to_tensor(actions,dtype=tf.int32)
    batch_actions=tf.one_hot(batch_actions,a_size,dtype=tf.float32)
    batch_states=tf.convert_to_tensor(states,dtype=tf.float32)
    batch_valids=tf.convert_to_tensor(valids,dtype=tf.float32)
    

    variables_for_actor=global_network.policy_dense1.trainable_variables+global_network.policy_dense2.trainable_variables+global_network.policy_dense3.trainable_variables
    actor_variable_names = set([v.name for v in variables_for_actor])
    all_trainable_variables = global_network.trainable_variables
    variables_except_for_actor = [
        v for v in all_trainable_variables 
        if v.name not in actor_variable_names
    ]

    
    world_grads,policy_grads,loss_list=tape_calc(global_network,batch_obs, batch_goals, batch_rewards, batch_actions, batch_states, batch_valids)

    var_norms = tf.linalg.global_norm(global_network.trainable_variables)

    world_grads, world_grad_norms = tf.clip_by_global_norm(world_grads, GRAD_CLIP)
    policy_grads, policy_grad_norms=tf.clip_by_global_norm(policy_grads, GRAD_CLIP )

    loss_list.append(world_grad_norms)
    loss_list.append(policy_grad_norms)
    loss_list.append(var_norms)
    
    
    world_optimizer.apply_gradients(zip(world_grads,variables_except_for_actor))
    policy_optimizer.apply_gradients(zip(policy_grads,variables_for_actor))
    
    if ADAPT_LR:
        lr = LR_Q / tf.sqrt(ADAPT_COEFF * curr_episode + 1.0)
        world_optimizer.learning_rate.assign(float(lr))
        policy_optimizer.learning_rate.assign(float(lr))
    else:
        world_optimizer.learning_rate.assign(LR_Q)
        policy_optimizer.learning_rate.assign(LR_Q)

    return loss_list
    
    

    





    

def writeToTensorBoard(global_summary, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric
    
    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.mean(tensorboardData, axis=0))

        rewardLoss, valueLoss, consistencyLoss, policyLoss, validLoss, entropy, worldgradNorm, policygradNorm, varNorm, \
            mean_length, mean_value, mean_invalid, \
            mean_stop, mean_astar,mean_collision, mean_reward, mean_finishes = tensorboardData
        
    else:
        firstEpisode = tensorboardData[0]
        rewardLoss, valueLoss, consistencyLoss, policyLoss, validLoss, entropy, worldgradNorm, policygradNorm, varNorm, \
            mean_length, mean_value, mean_invalid, \
            mean_stop, mean_astar,mean_collision, mean_reward, mean_finishes = firstEpisode

    global global_mean_finishes, global_10epi_finishes_0, global_10epi_finishes_1
    global_10epi_finishes_1=global_10epi_finishes_0
    global_10epi_finishes_0=mean_finishes
    global_mean_finishes=(global_10epi_finishes_0+global_10epi_finishes_1)/2.0

    with global_summary.as_default():
        tf.summary.scalar('Perf/Reward',mean_reward,curr_episode)
        tf.summary.scalar('Perf/Targets Done',mean_finishes,curr_episode)
        tf.summary.scalar('Perf/Length',mean_length,curr_episode)
        tf.summary.scalar('Perf/Valid Rate',(mean_length-mean_invalid)/mean_length,curr_episode)
        tf.summary.scalar('Perf/Stop Rate',mean_stop/mean_length,curr_episode)
        tf.summary.scalar('Perf/Astar Rate',mean_astar/mean_length,curr_episode)
        tf.summary.scalar('Perf/Collision Rate',mean_collision/mean_length,curr_episode)

        tf.summary.scalar('Losses/Reward Loss',rewardLoss,curr_episode)
        tf.summary.scalar('Losses/Value Loss',valueLoss,curr_episode)
        tf.summary.scalar('Losses/Consistency Loss',consistencyLoss,curr_episode)
        tf.summary.scalar('Losses/Policy Loss',policyLoss,curr_episode)
        tf.summary.scalar('Losses/Valid Loss',validLoss,curr_episode)
        tf.summary.scalar('Losses/Entropy',entropy,curr_episode)
        tf.summary.scalar('Losses/allGrad Norm',worldgradNorm,curr_episode)
        tf.summary.scalar('Losses/policyGrad Norm',policygradNorm,curr_episode)
        tf.summary.scalar('Losses/Var Norm',varNorm,curr_episode)
        global_summary.flush()



class ReplayBuffer():
    def __init__(self):
        os.makedirs("replay_buffer", exist_ok=True)

        # バッファの定義  [episode_num,step_num,--]
        self.obs_buffer = []
        self.goals_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.states_buffer = []
        self.valids_buffer = []

        self.indexlist = []

        self.iter = 0 #保有エピソード数
        self.startindex=0
        self.deletecount=0
        self.addcount=0

        if os.path.exists("replay_buffer/rb_data.pkl"):
            if load_model == True:
                with open("replay_buffer/rb_data.pkl",'rb') as f:
                    data=pickle.load(f)
                    self.obs_buffer=data["obs_buffer"]
                    self.goals_buffer=data["goals_buffer"]
                    self.actions_buffer=data["actions_buffer"]
                    self.rewards_buffer=data["rewards_buffer"]
                    self.states_buffer=data["states_buffer"]
                    self.valids_buffer=data["valids_buffer"]
                    self.indexlist=data["indexlist"]
                    
                    self.deletecount=data["deletecount"]
                    self.addcount=data["addcount"]

            

        self.iter=len(self.goals_buffer)
       


    def add(self, obs, goals, actions, rewards, states, valids):  #訓練が進みエピソードの長さが減る分バッファの保持ステップ数が減るのは問題かも？
        if self.iter >= replay_buffer_size:
            deleted_obs=self.obs_buffer.pop(0)
            self.obs_buffer.append(obs)
            self.goals_buffer.pop(0)
            self.goals_buffer.append(goals)
            self.actions_buffer.pop(0)
            self.actions_buffer.append(actions)
            self.rewards_buffer.pop(0)
            self.rewards_buffer.append(rewards)
            self.states_buffer.pop(0)
            self.states_buffer.append(states)
            self.valids_buffer.pop(0)
            self.valids_buffer.append(valids)

            """
            self.indexlist=[i for i in self.indexlist if i[0]!=0] #古いやつのインデックス候補消す
            for i in range(len(self.indexlist)):
                self.indexlist[i][0]-=1
            episode_length=len(obs)
            """
            for i in range(len(obs)-horizon):
                index=np.array([self.addcount,i])
                self.indexlist.append(index)

            self.indexlist=self.indexlist[len(deleted_obs)-horizon:]
            self.deletecount+=1
            self.addcount+=1
            

        else:
            self.obs_buffer.append(obs)
            self.goals_buffer.append(goals)
            self.actions_buffer.append(actions)
            self.rewards_buffer.append(rewards)
            self.states_buffer.append(states)
            self.valids_buffer.append(valids)

            for i in range(len(obs)-horizon):
                index=np.array([self.addcount,i])
                self.indexlist.append(index)
            self.iter += 1
            self.addcount+=1


    def sample(self, batch_size, horizon):

        rng = np.random.default_rng()
        r=rng.normal(0,len(self.indexlist)//10,batch_size)
        r=np.abs(r)
        r=np.clip(r,0,len(self.indexlist)-1)
        r=r.astype(int)

        sample_ids=-r+(len(self.indexlist)-1)
        

        # バッファからデータを取得
        obs = np.empty(( batch_size, horizon+1,11,11,11), dtype=np.float32)
        goals= np.empty((batch_size,horizon+1,3))
        actions = np.empty(( batch_size,horizon, ), dtype=np.float32)
        rewards = np.empty((batch_size,horizon,  ), dtype=np.float32)
        states = np.empty((batch_size,2,1,512),dtype=np.float32)
        valids = np.empty((batch_size,horizon,5),dtype=np.float32)

        for i in range(batch_size):
            obs[i]=np.stack(self.obs_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]:self.indexlist[sample_ids[i]][1]+horizon+1])
            goals[i]=np.stack(self.goals_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]:self.indexlist[sample_ids[i]][1]+horizon+1])
            actions[i]=np.stack(self.actions_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]:self.indexlist[sample_ids[i]][1]+horizon])
            rewards[i]=np.stack(self.rewards_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]:self.indexlist[sample_ids[i]][1]+horizon])
            states[i]=np.stack(self.states_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]])
            valids[i]=np.stack(self.valids_buffer[self.indexlist[sample_ids[i]][0]-self.deletecount][self.indexlist[sample_ids[i]][1]:self.indexlist[sample_ids[i]][1]+horizon])


        
        return obs,goals,actions,rewards,states,valids
    
    def save(self):
        with open("replay_buffer/rb_data.pkl",'wb') as f:
                data={}
                data["obs_buffer"]=self.obs_buffer
                data["goals_buffer"]=self.goals_buffer
                data["actions_buffer"]=self.actions_buffer
                data["rewards_buffer"]=self.rewards_buffer
                data["states_buffer"]=self.states_buffer
                data["valids_buffer"]=self.valids_buffer
                data["indexlist"]=self.indexlist
                
                data["deletecount"]=self.deletecount
                data["addcount"]=self.addcount
                pickle.dump(data, f)
        


    
def main():    
    with tf.device("/GPU:0"):
        world_optimizer = tf.keras.optimizers.Nadam(learning_rate=float(1))
        policy_optimizer= tf.keras.optimizers.Nadam(learning_rate=float(1))
        global_network = ACRDNet()

        #ダミーデータでのネットワーク構築
        dummy_obs=tf.zeros([1,1,11,11,11])
        dummy_goals=tf.zeros([1,1,3])   
        dummy_h_state=tf.zeros([1,512]) 
        dummy_c_state=tf.zeros([1,512])  
        dummy_latents=tf.zeros([1,1,512])
        dummy_actions=tf.constant([[[1.0, 0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)

        global_network.encode(dummy_obs,dummy_goals,dummy_h_state,dummy_c_state)
        global_network.dynamics(dummy_latents,dummy_actions) 
        global_network.policy(dummy_latents)
        global_network.reward(dummy_latents,dummy_actions)
        global_network.q1(dummy_latents,dummy_actions)
        global_network.q2(dummy_latents,dummy_actions)
        

        variables_for_actor=global_network.policy_dense1.trainable_variables+global_network.policy_dense2.trainable_variables+global_network.policy_dense3.trainable_variables
        actor_variable_names = set([v.name for v in variables_for_actor])
        all_trainable_variables = global_network.trainable_variables
        variables_except_for_actor = [
            v for v in all_trainable_variables 
            if v.name not in actor_variable_names
        ]

        dummy_world_grads = [tf.zeros_like(v) for v in variables_except_for_actor]
        dummy_policy_grads = [tf.zeros_like(v) for v in variables_for_actor]

        world_optimizer.apply_gradients(zip(dummy_world_grads, variables_except_for_actor))
        policy_optimizer.apply_gradients(zip(dummy_policy_grads, variables_for_actor))
     

        global_summary = tf.summary.create_file_writer(train_path)
        checkpoint = tf.train.Checkpoint(model=global_network, world_optimizer=world_optimizer,policy_optimizer=policy_optimizer)

        checkpoint_manager=tf.train.CheckpointManager(checkpoint,model_path,1)

   
    if load_model == True:
        print ('Loading Model...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

        p=checkpoint_manager.latest_checkpoint
        print("checkpoint name:",p)
        p=p[p.find('-')+1:]
        #p=p[:p.find('.')]
        curr_episode=int(p)

        print("curr_episode set to ",curr_episode)
    else:
        curr_episode = 0


    if ADAPT_LR:
        # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        # we need the +1 so that lr at step 0 is defined
        lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), curr_episode))))
    else:
        lr = tf.constant(LR_Q)
    world_optimizer.learning_rate.assign(float(lr))
    policy_optimizer.learning_rate.assign(float(lr))




    replaybuffer=ReplayBuffer()

    global global_mean_finishes
        
    
    # launch all of the threads:
    rl_agents = [RLRunner.remote(i, global_mean_finishes) for i in range(NUM_IL_META_AGENTS, NUM_META_AGENTS)]
    meta_agents = rl_agents 

    weights=global_network.get_weights()


    # launch the first job (e.g. getGradient) on each runner
    jobList = [] # Ray ObjectIDs 
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, curr_episode, global_mean_finishes))
        curr_episode += 1

    tensorboardData = []


    
    numRLEpisodes = 0
    try:
        while True:
            # wait for any job to be completed - unblock as soon as the earliest arrives
            done_id, jobList = ray.wait(jobList)
            
            # get the results of the task from the object store
            #jobResults, metrics, info = ray.get(done_id)[0]


            obsResults, goalsResults, actionsResults, rewardsResults, statesResults, validsResults, metrics, info= ray.get(done_id)[0]

            all_loss=[]
            
            if obsResults and goalsResults and actionsResults and rewardsResults and statesResults and validsResults:
                for i in range(len(obsResults)):
                    replaybuffer.add(obsResults[i],goalsResults[i],actionsResults[i],rewardsResults[i],statesResults[i],validsResults[i])
                if curr_episode>(random_term-2): #random_term個分が終わったタイミングから学習を始めたい。
                    for i in range(10):
                        obs,goals,actions,rewards,states,valids=replaybuffer.sample(batch_size,horizon)
                        loss_list=update(global_network,obs,goals,actions,rewards,states,valids,world_optimizer,policy_optimizer,curr_episode)
                        all_loss.append(loss_list)
                        print("update loop")
                    avg_loss=list(np.mean(np.array(all_loss), axis=0))
                    all_metrics=avg_loss+metrics
                else:
                    all_metrics=[0,0,0,0,0,0,0,0,0]+metrics
                tensorboardData.append(all_metrics)
                numRLEpisodes += 1


            
            


            # Every `SUMMARY_WINDOW` RL episodes, write RL episodes to tensorboard
            if len(tensorboardData) >= SUMMARY_WINDOW:
                writeToTensorBoard(global_summary, tensorboardData, curr_episode)
                tensorboardData = []
                
            # get the updated weights from the global network
            
            weights = global_network.get_weights()
            curr_episode += 1

            # start a new job on the recently completed agent with the updated weights
            jobList.extend([meta_agents[info['id']].job.remote(weights, curr_episode,global_mean_finishes)])

            
            if curr_episode % 100 == 0:
                print ('Saving Model', end='\n')
                #checkpoint_numberのところにエピソードナンバーを保存しておく
                checkpoint_manager.save(checkpoint_number=curr_episode)
                replaybuffer.save()
                print ('Saved Model', end='\n')

            
                
    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__": 
    main()
