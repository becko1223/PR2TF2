import tensorflow as tf
import scipy.signal as signal
import copy
import numpy as np
import ray
import os
import imageio
import random
from Env_Builder import *

from Map_Generator import maze_generator

from parameters import *

GRAD_CLIP = 10.0
RNN_SIZE = 512


# helper functions
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

@tf.function
def action2dir_tensor(a):
    checking_tensor = tf.constant([
        [0, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0]
    ], dtype=tf.float32)
    direction_tensor = tf.gather(checking_tensor, a)
    
    return direction_tensor



def distribution_to_coordinate(actions_distribution):
    checking_tensor = tf.constant([
        [0.0, 0.0], # 0: 待機
        [0.0, 1.0], # 1: 上
        [1.0, 0.0], # 2: 右
        [0.0, -1.0],# 3: 下
        [-1.0, 0.0] # 4: 左
    ], dtype=tf.float32)

    coord=tf.matmul(actions_distribution,checking_tensor)
    return coord



class Worker():
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent, env, localNetwork, groupLock,inferenceLock,mean_finishes, learningAgent,
                 ):

        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_workers = workers_per_metaAgent
        
        self.nextGIF = 0

        self.env = env
        self.local_ACRD = localNetwork
        self.groupLock = groupLock
        self.inferenceLock=inferenceLock
        self.mean_finisheds = mean_finishes
        self.learningAgent = learningAgent
        self.allGradients = []
        self.loss_metrics =[]
        self.perf_metrics= np.zeros(6)
        


   


        

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1, RNN_SIZE], dtype=tf.float32)
    ])
    def sample_from_actor(self,latent_inits):    #init:[batch,1,feature]

        """
        #tf.scanでやろうとしていたときのもの（これだと不安定で上手く動かない）
        def scan_fn(actions_latents,elem):
            policy_logits=tf.clip_by_value(self.local_ACRD.policy(actions_latents[1]),-10.0,10.0)
            print("policy_logits shape:",policy_logits.shape)
            action_probs=tf.nn.softmax(policy_logits)
            action_probs=tf.squeeze(action_probs,axis=1)
            
         
            #actions=tf.map_fn(lambda action_prob: np.random.choice(range(a_size),p=action_prob),action_probs)
            actions = tf.squeeze(tf.random.categorical(tf.math.log(action_probs), num_samples=1), axis=-1)
            actions_onehot=tf.one_hot(actions,a_size)
            actions_onehot=tf.expand_dims(actions_onehot,axis=1) #dynamics入力のため
            latent_preds=self.local_ACRD.dynamics(actions_latents[1],actions_onehot)
            print("actions_onehot shape:",actions_onehot.shape)
            return (actions_onehot,latent_preds)
        
        elems=range(0,horizon)
        batch_size = tf.shape(latent_inits)[0]
        init_action = tf.zeros([batch_size, 1, a_size], dtype=tf.float32)
        init = (init_action, init_latents)

        print("init[1] shape:",init[1].shape)
        result=tf.scan(fn=scan_fn,elems=elems,initializer=init)
        actions=result[0]  
        actions=tf.squeeze(actions,axis=2)
        actions=tf.transpose(actions,[1,0,2])
        print("actions shape:",actions.shape)

        """

       
        current_latent = latent_inits 
        
        actions_ta = tf.TensorArray(dtype=tf.float32, size=horizon, dynamic_size=False)

        for t in tf.range(horizon):
            
            
            # Policy (B, 1, A_SIZE)
            #print("current_latent shape:", current_latent.shape)
            policy_logits=self.local_ACRD.policy(current_latent)
            policy_logits = tf.clip_by_value(policy_logits, -10.0, 10.0)
            
            # Action Probabilities (B, A_SIZE)
            action_probs = tf.nn.softmax(policy_logits)
            action_probs = tf.squeeze(action_probs, axis=1)

            # Sample Action Index (B,)
            actions = tf.squeeze(tf.random.categorical(tf.math.log(action_probs), num_samples=1), axis=-1)
            
            # Action One-Hot (B, 1, A_SIZE)
            actions_onehot = tf.one_hot(actions, a_size)
            actions_onehot = tf.expand_dims(actions_onehot, axis=1) 
            
            # Predict Next Latent (B, 1, 512)
            current_latent = self.local_ACRD.dynamics(current_latent, actions_onehot)
            
            # 結果をTensorArrayに書き込む (B, A_SIZE)
            actions_ta = actions_ta.write(t, tf.squeeze(actions_onehot, axis=1))

        # TensorArrayから結果を取り出し、形状を整える
        actions = actions_ta.stack() # [horizon, B, A_SIZE]
        actions = tf.transpose(actions, [1, 0, 2]) # [B, horizon, A_SIZE]


        return actions

    


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[horizon, a_size], dtype=tf.float32),
        tf.TensorSpec(shape=[horizon,],dtype=tf.float32)
    ])
    def sample_from_distribution(self,actions_mean,actions_std):
        
       #[horizon,2]
        actions_mean_2D=distribution_to_coordinate(actions_mean)
        actions_mean_2D=tf.expand_dims(actions_mean_2D,0)
        
        actions_mean_2D_samples=tf.repeat(actions_mean_2D,num_samples,axis=0)
        eps=tf.random.normal([num_samples,horizon,2])
        actions_std=tf.expand_dims(actions_std,1)
        #print("actions_mean_2D_samples shape:",actions_mean_2D_samples.shape)
        #print("eps shape:",eps.shape)
        #print("std shape:",actions_std.shape)
        actions=actions_mean_2D_samples+actions_std*eps      #[B,horizon,2]
        
        """
        if(self.agentID==1):
            tf.print(
            "meta:", self.metaAgentID, 
            " agent:", self.agentID, 
            " one of sample:", actions[5][0],
            summarize=-1,  # summarize=-1 でテンソルの全要素を出力
            )
        """


        def coordinate_to_onehot(action):
            targets = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=tf.float32)
    
            action_expanded = tf.expand_dims(action, axis=0) # 形状: (1, 2)
            diff = targets - action_expanded 
            

            distance = tf.norm(diff, ord='euclidean', axis=1) # 形状: (5,)
            

            index = tf.argmin(distance)

            onehot = tf.one_hot(index, a_size) # 形状: (a_size,)
            return onehot
        

        #actions_onehot_samples=tf.map_fn(fn=lambda x:tf.map_fn(fn=coordinate_to_onehot,elems=x),elems=actions)
        # ターゲット座標: (A_SIZE, 2)
        targets = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=tf.float32)
        
        # 形状を合わせるために targets を拡張: (1, 1, A_SIZE, 2)
        targets_expanded = tf.expand_dims(tf.expand_dims(targets, 0), 0)

        # actions を拡張: (B, T, 1, 2)
        actions_expanded = tf.expand_dims(actions, axis=-2)
        
        # 差分を計算: (B, T, A_SIZE, 2)
        diff = targets_expanded - actions_expanded
        
        # 距離を計算: (B, T, A_SIZE)
        distance = tf.norm(diff, ord='euclidean', axis=-1) 
        
        # 最小距離のインデックス (アクションID) を取得: (B, T)
        index = tf.argmin(distance, axis=-1)

        # One-hotに変換: (B, T, A_SIZE)
        actions_onehot_samples = tf.one_hot(index, a_size)

        """
        targets = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=tf.float32)

        B = tf.shape(actions)[0]
        T = tf.shape(actions)[1]
        
 
        actions_onehot_ta = tf.TensorArray(dtype=actions.dtype, size=0, dynamic_size=True, clear_after_read=False)
        flat_index = 0 
        for i in tf.range(actions.shape[0]):
            for j in tf.range(actions.shape[1]):
                action=actions[i][j]
                action_expanded = tf.expand_dims(action, axis=0) # 形状: (1, 2)
                diff = targets - action_expanded 
                distance = tf.norm(diff, ord='euclidean', axis=1) # 形状: (5,)
                index = tf.argmin(distance)
                onehot = tf.one_hot(index, a_size) # 形状: (a_size,)
                actions_onehot_ta = actions_onehot_ta.write(flat_index, onehot)
                flat_index += 1


        actions_onehot_samples = actions_onehot_ta.stack()
        actions_onehot_samples = tf.reshape(actions_onehot_samples, (B, T, a_size))


        """

        #invalidなものを取り除くが、環境モデルのことを考えるとinvalidな選択肢を絶対に取らせないようにするのは良くないかも

        #first_step=actions_onehot_samples[:,0]

        #cond=tf.argmax(first_step) in validActions

        #actions_onehot_samples=tf.boolean_mask(actions_onehot_samples,cond)

        return actions_onehot_samples

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,horizon, a_size], dtype=tf.float32),
        tf.TensorSpec(shape=[None,1,512],dtype=tf.float32),
        tf.TensorSpec(shape=[None,a_size],dtype=tf.float32),
        tf.TensorSpec(shape=[1], dtype=tf.bool),
        tf.TensorSpec(shape=[2],dtype=tf.float32)
    ])
    def compute_return(self,samples,latent_inits,validActions,is_no_goalguide_condition,guide_dir): #[B,horizon,onehot] , [B,1,latentdim]

        """
        #scanはむずかしい
        def rollout(carry,actions):
            latents,discount,_=carry
            actions=tf.expand_dims(actions,axis=1)
            next_latents=self.local_ACRD.dynamics(latents,actions)  
            rewards=self.local_ACRD.reward(latents,actions)
            rewards=tf.squeeze(rewards,axis=1)
            return (next_latents, discount*gammma_tdmpc, discount*rewards)
        
        samples=tf.transpose(samples,[1,0,2]) #[B,S,5] to [S,B,5]
        
        latents,discounts,rewards=zip(*tf.scan(fn=rollout,elems=samples,initializer=(latent_inits,1.0,0.0))) #[S,B,1,latentsdim],[S,B,1(reward)]
        last_policies=self.local_ACRD.policy(latents[-1])
        last_policies=tf.clip_by_value(last_policies,-10,10)
        last_policies=tf.nn.softmax(last_policies)
        
        last_actions=tf.map_fn(lambda last_policy:tf.map_fn(lambda action_prob: np.random.choice(range(a_size),p=action_prob),last_policy),last_policies)  #ここは一個サンプリングよりも、前行動について確率重み付け平均撮った方が良いのかも。でも連続空間でやってる先行コードはこっち。
        last_actions=tf.one_hot(last_actions)
        q1_value=self.local_ACRD.q1(latents[-1],last_actions)
        q2_value=self.local_ACRD.q2(latents[-1],last_actions)
        q_value=tf.minimum(q1_value,q2_value)
        V=tf.reduce_sum(rewards,axis=0)+discounts[-1]*tf.squeeze(q_value,1)

        """
        B_size=tf.shape(samples)[0]
        current_latents=latent_inits
        samples=tf.transpose(samples,[1,0,2]) #[B,S,5] to [S,B,5]
        discount=1.0
        rewards_ta = tf.TensorArray(dtype=tf.float32, size=horizon, dynamic_size=False,clear_after_read=False)

        

        for t in tf.range(horizon):
            actions=samples[t]
            actions_expanded = tf.expand_dims(actions, axis=1)

            #print("actions shape:",actions.shape)
            rewards=self.local_ACRD.reward(current_latents,actions_expanded)
            rewards=tf.squeeze(tf.squeeze(rewards,axis=1),axis=1)

            #tf.print("penalty_tensor shape:",tf.shape(penalty_tensor))
            
            current_latents=self.local_ACRD.dynamics(current_latents,actions_expanded)
            rewards_ta=rewards_ta.write(t,rewards*discount)
            discount*=gammma_tdmpc


        last_policies=self.local_ACRD.policy(current_latents)
        last_policies=tf.clip_by_value(last_policies,-10,10)
        last_policies=tf.nn.softmax(last_policies)
        
        #last_actions=tf.map_fn(lambda last_policy: tf.random.categorical(tf.math.log(last_policy), num_samples=1),last_policies,dtype=tf.int64)  #ここは一個サンプリングよりも、前行動について確率重み付け平均撮った方が良いのかも。でも連続空間でやってる先行コードはこっち。
        policy_for_sampling = tf.squeeze(last_policies, axis=1) # [B, A_SIZE]

        """
        # バッチ全体でサンプリングを実行 (num_samples=1)
        last_actions = tf.random.categorical(
            tf.math.log(policy_for_sampling), 
            num_samples=1, 
            dtype=tf.int64
        )   #[B,1]
        """

        q_expected=tf.zeros([B_size,1,1],dtype=tf.float32)

        for t in tf.range(a_size):
            actions_expanded=tf.expand_dims(tf.repeat(tf.expand_dims(tf.one_hot(t,a_size),axis=0),B_size,axis=0),axis=1)
            q1_value=self.local_ACRD.q1(current_latents,actions_expanded)
            q2_value=self.local_ACRD.q2(current_latents,actions_expanded)
            q_value=tf.minimum(q1_value,q2_value)
            q_expected+=tf.expand_dims(tf.expand_dims(policy_for_sampling[:, t], axis=1), axis=2)*q_value


        #last_actions=tf.squeeze(last_actions)
        #last_actions = tf.one_hot(last_actions, a_size)
        #print("last_actions shape:",last_actions.shape)
        #last_actions=tf.expand_dims(last_actions,axis=1)
        #q1_value=self.local_ACRD.q1(current_latents,last_actions)
        #q2_value=self.local_ACRD.q2(current_latents,last_actions)
        #q_value=tf.minimum(q1_value,q2_value)

        rewards=rewards_ta.stack()
        #print("rewards shape:",rewards.shape)
        V=tf.reduce_sum(rewards,axis=0)+discount*tf.squeeze(tf.squeeze(q_expected,1),1)   


        #print("V shape:",V.shape)
        V=tf.squeeze(V) #[512,1]to[512,]
        #tf.print("V shape after tf.squeeze(V):", tf.shape(V))


        #系列評価ペナルティ
        wait_action = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        def compute_penalty(B_bool, penalty_weight):
            penalty_tensor = tf.cast(B_bool, dtype=tf.float32) * penalty_weight
            return penalty_tensor
        is_guide_term_condition = tf.less(self.currEpisode, guide_term)

        actions=samples[0]
        actions_expanded = tf.expand_dims(actions, axis=1)

        is_wait_action = tf.reduce_all(tf.equal(actions, wait_action), axis=1) # shape [B] (boolean)

        validActions_expanded = tf.expand_dims(validActions, axis=0) 
        validActions_tiled = tf.tile(validActions_expanded, [B_size, 1, 1]) # [B, N, A_SIZE]
        all_equal_to_valid = tf.reduce_all(tf.equal(actions_expanded, validActions_tiled), axis=2) # [B, N]
        is_valid_action = tf.reduce_any(all_equal_to_valid, axis=1)
        is_invalid_action = tf.logical_not(is_valid_action)

        actions_dir=distribution_to_coordinate(actions)
        distance_from_goalguide=tf.math.reduce_euclidean_norm(actions_dir-guide_dir,axis=1)


        penalty_tensor = tf.cond(is_guide_term_condition, 
            lambda: tf.cond(is_no_goalguide_condition,
                            lambda: compute_penalty(is_wait_action, -0.1)+compute_penalty(is_invalid_action,-0.5) , 
                            lambda: compute_penalty(is_wait_action, -0.1)+compute_penalty(is_invalid_action,-0.5)-distance_from_goalguide*0.25),
            lambda: tf.zeros_like(V) 
        )

        V+=penalty_tensor*max([0.0,((20.0-self.mean_finisheds)/20.0)])
        
        return V


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, ], dtype=tf.float32),
        tf.TensorSpec(shape=[None,horizon,a_size],dtype=tf.float32)
    ])
    def get_mean(self,V,samples):

        topK=tf.math.top_k(V,k=num_elites)
        V_elite=topK.values                #[k,]
        actions_elite=tf.gather(samples, topK.indices) #[k,horizon,5]
        score=tf.math.exp(temperature * (V_elite - tf.reduce_max(V_elite)))
        score=score/(tf.reduce_sum(score)+ 1e-9)   #[k,(score)]

        score=tf.expand_dims(score,axis=1)
        score=tf.expand_dims(score,axis=1)  #[k,1,1]mean計算のために

        #print("actions_elite shape:",actions_elite.shape)
        #print("score shape:",score.shape)
        
        mean=tf.reduce_mean(actions_elite*score,axis=0)   #[horizon,5]

        action_indices = tf.argmax(actions_elite, axis=-1, output_type=tf.int32) # [k, horizon]
        elite_coord = action2dir_tensor(action_indices) 
        #elite_coord=tf.map_fn(fn=lambda x:tf.map_fn(fn=onehot_to_coordinate,elems=x),elems=actions_elite)
        mean_coord=distribution_to_coordinate(mean)

        #print("score shape:",score.shape)
        #print("elite_coord shape:",elite_coord.shape)
        #print("mean coord shape",mean_coord.shape)
        score=tf.squeeze(score,axis=-1) #[k,1]に戻す。次の行の計算のため
        batch_for_std=score * tf.math.reduce_euclidean_norm(elite_coord - mean_coord,axis=-1)**2
        #print("batch_for_std shape:",batch_for_std.shape)
        std=tf.sqrt(tf.reduce_sum(batch_for_std,axis=0))
        #print("std shape",std.shape)

        return mean, std


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1,1, 512], dtype=tf.float32),
        tf.TensorSpec(shape=[horizon,a_size],dtype=tf.float32),
        tf.TensorSpec(shape=[None,a_size],dtype=tf.float32),
        tf.TensorSpec(shape=[1],dtype=tf.bool),
        tf.TensorSpec(shape=[2],dtype=tf.float32)
    ], reduce_retracing=True)
    def mppi(self,latent_init,mean,validActions,is_no_guide,guide_dir):
        std=tf.ones([horizon,])

        #print("latent_init shape:",latent_init.shape)
        inits_for_actor=tf.repeat(latent_init,num_actor_traj,axis=0)
        
        inits_for_return=tf.repeat(latent_init,num_actor_traj+num_samples,axis=0)
        

        samples_from_actor=self.sample_from_actor(inits_for_actor)        #[B,horizon,onehot]

       


        for i in tf.range(iterations):
            samples_from_distribution=self.sample_from_distribution(mean,std) 
            allsamples=tf.concat([samples_from_actor,samples_from_distribution],axis=0)
            V=self.compute_return(allsamples,inits_for_return,validActions,is_no_guide,guide_dir)
            #tf.print("V shape befre get_mean:", tf.shape(V))
            mean,std=self.get_mean(V,allsamples)

        samples_from_distribution=self.sample_from_distribution(mean,std)
        inits_for_return=tf.repeat(latent_init,num_samples,axis=0)
        V=self.compute_return(samples_from_distribution,inits_for_return,validActions,is_no_guide,guide_dir)

        action_best=tf.argmax(samples_from_distribution[tf.argmax(V),0])

        return action_best,mean







    def calculateImitationGradient(self, rollout, episode_count):
        rollout = np.array(rollout, dtype=object)
        # we calculate the loss differently for imitation
        # if imitation=True the rollout is assumed to have different dimensions:
        # [o[0],o[1],optimal_actions]

        rnn_state = [self.local_AC.h0,self.local_AC.c0]
        
        with tf.GradientTape() as tape:
            with self.inferenceLock:
                latent,_=self.local_ACRD.encode(np.expand_dims(np.stack(rollout[:, 0]),0),np.expand_dims(np.stack(rollout[:, -4]),0),np.expand_dims(rnn_state))
                policy=self.local_ACRD.policy(latent)
            policy=tf.clip_by_value(policy,-10,10)
            policy=tf.nn.softmax(policy)

            optimal_actions_onehot = tf.one_hot(np.expand_dims(np.stack(rollout[:, 2]),axis=0), a_size, dtype=tf.float32)

            loss=tf.reduce_mean(tf.keras.backend.categorical_crossentropy(optimal_actions_onehot, policy))

        i_grads = tape.gradient(loss,self.local_AC.trainable_variables)


        return [loss], i_grads

    

    def calculateGradient(self, rollout, episode_count, rnn_state0):
        
        rollout = np.array(rollout, dtype=object)
        obs=np.stack(rollout[:, 0])
        goals=np.stack(rollout[:,-4])
        rewards = np.stack(rollout[:, 2])
        actions = np.stack(rollout[:, 1])
        
        rnn_states=np.stack(rollout[:,-1])
        valids = np.stack(rollout[:, 4])

       

        

        

        #rewards_array = np.array([float(r) for r in rewards])
        #self.rewards_plus = np.concatenate([rewards_array, [bootstrap_value]])
        #discounted_rewards = discount(rewards_array, gamma)[:-1]
        #discounted_rewards=tf.convert_to_tensor(discounted_rewards)
        


        #エピソードの切り分け
        step=len(rollout)
        all_list=range(0,step-(horizon))
        batch_size=(step//horizon)
        if(step<256):               #ゴール報酬経験の訓練優先
            batch_size*=3
            chosen=random.choices(all_list,k=batch_size)
            for t in tf.range(10):
                chosen.append(step-horizon-1)
                batch_size+=1
        else:
            chosen=random.choices(all_list,k=batch_size)
        
            
            
            

        np_obs=np.stack([obs[i:i+horizon+1] for i in chosen])  #長さhorizon+1
        np_goals=np.stack([goals[i:i+horizon+1] for i in chosen])
        np_rewards=np.stack([rewards[i:i+horizon+1] for i in chosen])
        np_actions=np.stack([actions[i:i+horizon+1] for i in chosen])
        np_states=np.stack([rnn_states[i] for i in chosen])
        np_valids=np.stack([valids[i:i+horizon+1] for i in chosen])

        batch_obs = tf.convert_to_tensor(np_obs,dtype=tf.float32) 
        batch_goals = tf.convert_to_tensor(np_goals,dtype=tf.float32)
        batch_rewards=tf.convert_to_tensor(np_rewards,dtype=tf.float32)
        #batch_discounted_rewards = tf.stack([discounted_rewards[i:i+horizon+1] for i in chosen])
        batch_actions=tf.convert_to_tensor(np_actions,dtype=tf.int32)
        batch_actions=tf.one_hot(batch_actions,a_size,dtype=tf.float32)
        #batch_train_value=tf.stack([train_value[i:i+horizon+1] for i in chosen])
        batch_states=tf.convert_to_tensor(np_states,dtype=tf.float32)
        batch_valids=tf.convert_to_tensor(np_valids,dtype=tf.float32)
        rhos=tf.convert_to_tensor([[rho**i for i in range(horizon)] for j in chosen])

        @tf.function
        def tape_calc(self,batch_obs, batch_goals, batch_rewards, batch_actions, batch_states, batch_valids):
            variables_for_actor=self.local_ACRD.policy_dense1.trainable_variables+self.local_ACRD.policy_dense2.trainable_variables+self.local_ACRD.policy_dense3.trainable_variables
            actor_variable_names = set([v.name for v in variables_for_actor])
            all_trainable_variables = self.local_ACRD.trainable_variables
            variables_except_for_actor = [
                v for v in all_trainable_variables 
                if v.name not in actor_variable_names
            ]

            #アクター以外訓練
            with tf.GradientTape() as tape:

                def dynamics(carry, elem):
                    elem=tf.expand_dims(elem,axis=1)
                    prev_latents,_=carry
                    with self.inferenceLock:
                        latents = self.local_ACRD.dynamics(prev_latents,elem)
                        rewards= self.local_ACRD.reward(prev_latents,elem)
                    return (latents,rewards)
                
                batch_actions_T = tf.transpose(batch_actions[:, :-1], [1, 0, 2])  # [horizon, batch, action_dim]

                with self.inferenceLock:
                    latent_init,batch_states_step1=self.local_ACRD.encode(batch_obs[:, 0:1],batch_goals[:,0:1],tf.reshape(batch_states[:,0],[-1,512]),tf.reshape(batch_states[:,1],[-1,512]))

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
                batch_reward_preds=tf.squeeze(batch_reward_preds)
            
                #valueの予測値を出す
                with self.inferenceLock:
                    batch_q1value_preds=self.local_ACRD.q1(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],batch_actions[:,:-1])
                    batch_q2value_preds=self.local_ACRD.q2(tf.concat([latent_init,batch_latent_preds],axis=1)[:,:-1],batch_actions[:,:-1])
                    batch_q1value_preds=tf.squeeze(batch_q1value_preds)
                    batch_q2value_preds=tf.squeeze(batch_q2value_preds)


                #latentのターゲットを出す(b,s,h,w,c)
                with self.inferenceLock:
                    batch_latent_targets,_=self.local_ACRD.encode(batch_obs[:,1:],batch_goals[:,1:],batch_states_step1[0],batch_states_step1[1])

                #valueのターゲットを出す  一個行動抜き出してvalue出すか、各行動ごとの確率重み付け平均にするか悩む
                with self.inferenceLock:
                    policy=self.local_ACRD.policy(batch_latent_preds) #[B,H,a_size]
                    policy=tf.clip_by_value(policy,-10.0,10.0)
                    policy=tf.nn.softmax(policy)
                
                #next_actions=tf.map_fn(lambda probs: tf.random.categorical(probs, 1),elems=logits,dtype=tf.int64)   
                logits = tf.math.log(policy + 1e-10) # ゼロ除算を防ぐために微小値を加算
                B = tf.shape(logits)[0]
                H = tf.shape(logits)[1]
                A = tf.shape(logits)[2]
                flat_logits = tf.reshape(logits, [-1, A])
                flat_actions = tf.random.categorical(flat_logits, num_samples=1, dtype=tf.int64)
                next_actions = tf.reshape(flat_actions, [B, H, 1])
                next_actions = tf.squeeze(next_actions, axis=-1)
                next_actions=tf.one_hot(next_actions,a_size)
                with self.inferenceLock:
                    q1_next=self.local_ACRD.q1(batch_latent_preds,next_actions)
                    q2_next=self.local_ACRD.q2(batch_latent_preds,next_actions)
                q_next=tf.minimum(q1_next,q2_next)
                q_next=tf.squeeze(q_next)
                q_target=batch_rewards[:,1:]+gammma_tdmpc*q_next
                

                reward_loss=tf.reduce_mean(rhos*tf.square(batch_reward_preds-batch_rewards[:,1:]))
                q1value_loss=tf.reduce_mean(rhos*tf.square(q_target-batch_q1value_preds))
                q2value_loss=tf.reduce_mean(rhos*tf.square(q_target-batch_q2value_preds))
                consistency_loss=tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*tf.square(batch_latent_targets-batch_latent_preds))

                total_loss=0.5*reward_loss+0.1*(q1value_loss+q2value_loss)+2.0*consistency_loss
            with self.inferenceLock:
                world_grads=tape.gradient(total_loss,variables_except_for_actor)


            #アクター訓練
            with tf.GradientTape() as tape:
                
                with self.inferenceLock:
                    policy=self.local_ACRD.policy(batch_latent_preds)
                policy=tf.clip_by_value(policy,-10.0,10.0)
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
                with self.inferenceLock:
                    q1_next=self.local_ACRD.q1(batch_latent_preds,next_actions)
                    q2_next=self.local_ACRD.q2(batch_latent_preds,next_actions)
                batch_q=tf.minimum(q1_next,q2_next)
                batch_q=tf.squeeze(batch_q)
                
                batch_policies_sig=tf.sigmoid(policy)


                policy_loss=-tf.reduce_mean(rhos*batch_q)
                
                valid_loss=-tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*(batch_valids[:,1:]*tf.math.log(tf.clip_by_value(batch_policies_sig, 1e-10, 1.0))+(1-batch_valids[:,1:])*tf.math.log(tf.clip_by_value(1-batch_policies_sig,1e-10,1.0))))
                entropy=-tf.reduce_mean(tf.expand_dims(rhos,axis=-1)*policy * tf.math.log(tf.clip_by_value(policy, 1e-10, 1.0)))

                total_loss=0.5*policy_loss+16*valid_loss+entropy
            with self.inferenceLock:
                policy_grads=tape.gradient(total_loss,variables_for_actor)
            return world_grads,policy_grads,reward_loss,q1value_loss,q2value_loss,consistency_loss,policy_loss,valid_loss,entropy
        

        world_grads,policy_grads,reward_loss,q1value_loss,q2value_loss,consistency_loss,policy_loss,valid_loss,entropy=tape_calc(self,batch_obs, batch_goals, batch_rewards, batch_actions, batch_states, batch_valids)


        var_norms = tf.linalg.global_norm(self.local_ACRD.trainable_variables)

        world_grads, world_grad_norms = tf.clip_by_global_norm(world_grads, GRAD_CLIP)
        policy_grads, policy_grad_norms=tf.clip_by_global_norm(policy_grads, GRAD_CLIP )

        return [reward_loss,q1value_loss+q2value_loss,consistency_loss,policy_loss,valid_loss,entropy,world_grad_norms, policy_grad_norms, var_norms],world_grads,policy_grads



    def imitation_learning_only(self, episode_count):
        self.env._reset()
        rollouts, targets_done = self.parse_path(episode_count)

        if rollouts is None:
            return None, 0

        gradients = []
        losses = []
        for i in range(self.num_workers):
            train_buffer = rollouts[i]
            
            imitation_loss, grads = self.calculateImitationGradient(train_buffer, episode_count)

            gradients.append(grads)
            losses.append(imitation_loss)

        return gradients, losses

    def run_episode_multithreaded(self, episode_count, coord):

        if self.metaAgentID < NUM_IL_META_AGENTS:
            assert (1 == 0)
            # print("THIS CODE SHOULD NOT TRIGGER")
            self.is_imitation = True
            self.imitation_learning_only()

        global episode_lengths, episode_mean_values, episode_invalid_ops, episode_stop_ops, episode_rewards, episode_finishes

        num_agents = self.num_workers

        
        while self.shouldRun(coord, episode_count):
            episode_buffer, episode_values = [], []
            episode_reward = episode_step_count = episode_inv_count = targets_done = episode_stop_count = episode_astar_count= episode_collision_count= 0

            # Initial state from the environment
            if self.agentID == 1:
                self.env._reset()
                joint_observations[self.metaAgentID] = self.env._observe()

            self.synchronize()  # synchronize starting time of the threads

            # Get Information For Each Agent 
            validActions = self.env.listValidActions(self.agentID,
                                                        joint_observations[self.metaAgentID][self.agentID])

            s = joint_observations[self.metaAgentID][self.agentID]

            h_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
            c_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
            rnn_state = [h_init, c_init]
            rnn_state0 = rnn_state

            is_first_step=True

            pred_latent = tf.zeros([1,1,RNN_SIZE], dtype=tf.float32)

            mean=tf.one_hot(tf.zeros([horizon],dtype=tf.int32),a_size)

            self.synchronize()  # synchronize starting time of the threads
            swarm_reward[self.metaAgentID] = 0
            swarm_targets[self.metaAgentID] = 0

            episode_rewards[self.metaAgentID] = []
            episode_finishes[self.metaAgentID] = []
            episode_lengths[self.metaAgentID] = []
            episode_mean_values[self.metaAgentID] = []
            episode_invalid_ops[self.metaAgentID] = []
            episode_stop_ops[self.metaAgentID] = []

            # ===============================start training =======================================================================
            # RL
            if True:
                # prepare to save GIF
                saveGIF = False
                global GIFS_FREQUENCY_RL
                if OUTPUT_GIFS and self.agentID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF = episode_count + GIFS_FREQUENCY_RL
                    GIF_episode = int(episode_count)
                    GIF_frames = [self.env._render()]

                # start RL
                self.env.finished = False
                while not self.env.finished:


                    ob=tf.expand_dims(s[0],0)
                    ob=tf.expand_dims(ob,0)
                    ob=tf.cast(ob,dtype=tf.float32)
                    goal=tf.expand_dims(s[1],0)
                    goal=tf.expand_dims(goal,0)
                    goal=tf.cast(goal,dtype=tf.float32)

                    #print("ob shape:",ob.shape)
                    #print("goal shape:",goal.shape)
                    try:
                        tf.ensure_shape(rnn_state[0],[1,512])
                        tf.ensure_shape(rnn_state[1],[1,512])
                    except Exception as e:
                        print("state ensure error, step:",episode_step_count)
                        h_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
                        c_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
                        rnn_state = [h_init, c_init]

                    try:
                        tf.ensure_shape(rnn_state[0],[1,512])
                        tf.ensure_shape(rnn_state[1],[1,512])
                    except Exception as e:
                        print("state ensure error2")
                        h_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
                        c_init = tf.zeros([1, RNN_SIZE], dtype=tf.float32)
                        rnn_state = [h_init, c_init]


                    try:
                        tf.ensure_shape(mean,[horizon,a_size])
                    except Exception as e:
                        print("mean ensure error, step:",episode_step_count)
                        mean=tf.one_hot(tf.zeros([horizon],dtype=tf.int32),a_size)

                    try:
                        tf.ensure_shape(mean,[horizon,a_size])
                    except Exception as e:
                        print("mean ensure error2")
                        mean=tf.one_hot(tf.zeros([horizon],dtype=tf.int32),a_size)



                    latent_init,rnn_state=self.local_ACRD.encode(ob,goal,rnn_state[0],rnn_state[1])

                    model_error=tf.reduce_mean(tf.square(latent_init-pred_latent)) if not(is_first_step) else tf.constant([0.0])
                    is_first_step=False

                    is_no_guide=tf.constant([False],tf.bool)
                    guide_dir=tf.zeros([2],dtype=tf.float32)
                    astar_map=s[0][4]
                    if(random.random()<0.0001):
                        print(astar_map)
                    distance_list=[]
                    distance_list.append(1000) #待機が最短経路になることはない
                    distance_list.append(astar_map[4,5])
                    distance_list.append(astar_map[5,6])
                    distance_list.append(astar_map[6,5])
                    distance_list.append(astar_map[5,4])
                    distance_list_replace=[1000 if i<0 else i for i in distance_list]
                    if(distance_list_replace.count(0)>1):
                        is_no_guide=tf.constant([True],tf.bool)
                    else:
                        a_guide=distance_list_replace.index(min(distance_list_replace))
                        guide_dir=action2dir(a_guide)


                    rnn_state=[rnn_state[0],rnn_state[1]]


                    if(episode_count>random_term):
                        if(random.random()<0.1*((guide_term-episode_count)/guide_term-random_term)):
                            probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
                            indices = np.arange(len(probabilities))
                            a=np.random.choice(indices, p=probabilities)
                            
                        else:
                            validActions_onehot=tf.one_hot(tf.convert_to_tensor(np.array(validActions),dtype=tf.int32),a_size)
                            a, mean=self.mppi(latent_init,mean,validActions_onehot,is_no_guide,guide_dir)
                            a=a.numpy().item()
                        q=self.local_ACRD.q1(latent_init,tf.expand_dims(tf.expand_dims(tf.one_hot(a,a_size),0),0))
                        q=q.numpy()
                        mean=tf.concat([mean[1:],tf.one_hot(tf.constant([0]),a_size)],axis=0)
                       

                    else:
                        probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
                        indices = np.arange(len(probabilities))
                        a=np.random.choice(indices, p=probabilities)
                        q=np.zeros((1,1))

                    a_onehot=tf.one_hot(a,a_size)
                    a_onehot=tf.expand_dims(tf.expand_dims(a_onehot,axis=0),axis=0)
                    pred_latent=self.local_ACRD.dynamics(latent_init,a_onehot)
                    

                   

                    skipping_state = False
                    train_policy = train_val = 1

                    if not skipping_state:
                        '''
                        if not (np.argmax(tf.reshape(a_dist, [-1])) in validActions):
                            episode_inv_count += 1
                            train_val = 0  #最大行動がinvalidの場合valueを訓練しないのは、最大行動以外の行動で得た(というより、方策に従わずに得た)データはクリティックの訓練に不適切ということ？　状態価値でやるか行動価値でやるかによってもこれの必要性は変わる？
                        '''
                        if not (a in validActions):
                            episode_inv_count += 1
                        train_valid = np.zeros(a_size)
                        train_valid[validActions] = 1

                        

                        joint_actions[self.metaAgentID][self.agentID] = a
                        if a == 0:
                            episode_stop_count += 1

                        if not(is_no_guide):
                            if a==a_guide:
                                episode_astar_count += 1

                    # Make A Single Agent Gather All Information

                    self.synchronize()

                    if self.agentID == 1:
                        print("metaID:",self.metaAgentID," step",episode_step_count,"  agent1 action:",a)
                        all_obs, all_rewards = self.env.step_all(joint_actions[self.metaAgentID])
                        for i in range(1, self.num_workers + 1):
                            joint_observations[self.metaAgentID][i] = all_obs[i]
                            joint_rewards[self.metaAgentID][i] = all_rewards[i]
                            joint_done[self.metaAgentID][i] = (self.env.world.agents[i].status == 1)
                        if saveGIF and self.agentID == 1:
                            GIF_frames.append(self.env._render())

                    self.synchronize()  # synchronize threads

                    # Get observation,reward, valid actions for each agent 
                    s1 = joint_observations[self.metaAgentID][self.agentID]
                    error_reward=min([0.5*float(model_error.numpy()),0.05]) 
                    if(joint_rewards[self.metaAgentID][self.agentID]==-2.3):
                        episode_collision_count+=1
                    r = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID])+error_reward
                    validActions = self.env.listValidActions(self.agentID, s1)

                    self.synchronize()
                    # Append to Appropriate buffers 
                    if not skipping_state:
                        episode_buffer.append(
                            [s[0], a, joint_rewards[self.metaAgentID][self.agentID]+error_reward, s1, train_valid, s[1],
                                train_val, train_policy, rnn_state])
                        episode_values.append(q[0, 0])
                    episode_reward += r
                    episode_step_count += 1

                    # Update State
                    s = s1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if (len(episode_buffer) > horizon) and (
                            (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0) or joint_done[self.metaAgentID][
                        self.agentID] or episode_step_count == max_episode_length):
                        # Since we don't know what the true final return is,
                        # we "bootstrap" from our current value estimation.
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                            train_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                        else:
                            train_buffer = episode_buffer[:]

                        if joint_done[self.metaAgentID][self.agentID]:
                            s1Value = 0  # Terminal state
                            episode_buffer = []
                            joint_done[self.metaAgentID][self.agentID] = False
                            targets_done += 1
                            pred_latent=tf.zeros([1,1,RNN_SIZE], dtype=tf.float32)

                        #else:
                            
                            
                            
                            #latent_init,_=self.local_ACRD.encode(s[0],s[1],rnn_state)
                            
                            #for i in range(a_size):
                            #    s1Value+=a_dist[i]*self.local_ACRD.q1(latent_init,tf.expand_dims(tf.expand_dims(tf.one_hot(i)),0),0)
                            

                        
                        self.loss_metrics, world_grads, policy_grads = self.calculateGradient(train_buffer, episode_count,
                                                                            rnn_state0)

                        grads=(world_grads,policy_grads)
                        self.allGradients.append(grads)

                        rnn_state0 = rnn_state

                    self.synchronize()

                    # finish condition: reach max-len or all agents are done under one-shot mode
                    if episode_step_count >= max_episode_length:
                        break

                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                episode_stop_ops[self.metaAgentID].append(episode_stop_count)
                swarm_reward[self.metaAgentID] += episode_reward
                swarm_targets[self.metaAgentID] += targets_done

                self.synchronize()
                if self.agentID == 1:
                    episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])
                    episode_finishes[self.metaAgentID].append(swarm_targets[self.metaAgentID])

                    if saveGIF:
                        make_gif(np.array(GIF_frames),
                                    '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode,
                                                                            episode_step_count,
                                                                            swarm_reward[self.metaAgentID]))

                self.synchronize()

                perf_metrics = np.array([
                    episode_step_count,
                    np.nanmean(episode_values),
                    episode_inv_count,
                    episode_stop_count,
                    episode_astar_count,
                    episode_collision_count,
                    episode_reward,
                    targets_done
                ])

                assert len(self.allGradients) > 0, 'Empty gradients at end of RL episode?!'
                return perf_metrics

    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool

    def work(self, currEpisode, coord):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode

        if COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            self.perf_metrics = self.run_episode_multithreaded(currEpisode, coord)
        else:
            print("not implemented")
            assert (1 == 0)

            # gradients are accessed by the runner in self.allGradients
        return

    # Used for imitation learning
    def parse_path(self, episode_count):
        """needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)"""

        result = [[] for i in range(self.num_workers)]
        actions = {}
        o = {}
        train_imitation = {}
        targets_done = 0
        saveGIF = False

        if np.random.rand() < IL_GIF_PROB:
            saveGIF = True
        if saveGIF and OUTPUT_IL_GIFS:
            GIF_frames = [self.env._render()]

        single_done = False
        new_call = False
        new_MSTAR_call = False

        all_obs = self.env._observe()
        for agentID in range(1, self.num_workers + 1):
            o[agentID] = all_obs[agentID]
            train_imitation[agentID] = 1
        step_count = 0
        while step_count <= IL_MAX_EP_LENGTH:
            path = self.env.expert_until_first_goal()
            if path is None:  # solution not exists
                if step_count != 0:
                    return result, targets_done
                # print('Failed intially')
                return None, 0
            none_on_goal = True
            path_step = 1
            while none_on_goal and step_count <= IL_MAX_EP_LENGTH:
                completed_agents = []
                start_positions = []
                goals = []
                for i in range(self.num_workers):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    diff = tuple_minus(next_pos, self.env.world.getPos(agent_id))
                    actions[agent_id] = dir2action(diff)

                all_obs, _ = self.env.step_all(actions)
                for i in range(self.num_workers):
                    agent_id = i + 1
                    result[i].append([o[agent_id][0], o[agent_id][1], actions[agent_id], train_imitation[agent_id]])
                    if self.env.world.agents[agent_id].status == 1:
                        completed_agents.append(i)
                        targets_done += 1
                        single_done = True
                        if targets_done % MSTAR_CALL_FREQUENCY == 0:
                            new_MSTAR_call = True
                        else:
                            new_call = True
                if saveGIF and OUTPUT_IL_GIFS:
                    GIF_frames.append(self.env._render())
                if single_done and new_MSTAR_call:
                    path = self.env.expert_until_first_goal()
                    if path is None:
                        return result, targets_done
                    path_step = 0
                elif single_done and new_call:
                    path = path[path_step:]
                    path = [list(state) for state in path]
                    for finished_agent in completed_agents:
                        path = merge_plans(path, [None] * len(path), finished_agent)
                    try:
                        while path[-1] == path[-2]:
                            path = path[:-1]
                    except:
                        assert (len(path) <= 2)
                    start_positions_dir = self.env.getPositions()
                    goals_dir = self.env.getGoals()
                    for i in range(1, self.env.world.num_agents + 1):
                        start_positions.append(start_positions_dir[i])
                        goals.append(goals_dir[i])
                    world = self.env.getObstacleMap()
                    # print('OLD PATH', path) # print('CURRENT POSITIONS', start_positions) # print('CURRENT GOALS',goals) # print('WORLD',world)
                    try:
                        path = priority_planner(world, tuple(start_positions), tuple(goals), path)
                    except:
                        path = self.env.expert_until_first_goal()
                        if path is None:
                            return result, targets_done
                    path_step = 0
                o = all_obs
                step_count += 1
                path_step += 1
                new_call = False
                new_MSTAR_call = False
        if saveGIF and OUTPUT_IL_GIFS:
            make_gif(np.array(GIF_frames),
                     '{}/episodeIL_{}.gif'.format(gifs_path, episode_count))
        return result, targets_done

    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()
