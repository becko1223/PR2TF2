import tensorflow as tf
import scipy.signal as signal
import copy
import numpy as np
import ray
import os
import imageio
from Env_Builder import *

from Map_Generator import maze_generator

from parameters import *

GRAD_CLIP = 10.0


# helper functions
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent, env, localNetwork, groupLock, learningAgent,
                 ):

        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_workers = workers_per_metaAgent
        
        self.nextGIF = 0

        self.env = env
        self.local_AC = localNetwork
        self.groupLock = groupLock
        self.learningAgent = learningAgent
        self.allGradients = []
        self.loss_metrics =[]
        self.perf_metrics= np.zeros((1,))

    def calculateImitationGradient(self, rollout, episode_count):
        rollout = np.array(rollout, dtype=object)
        # we calculate the loss differently for imitation
        # if imitation=True the rollout is assumed to have different dimensions:
        # [o[0],o[1],optimal_actions]

        rnn_state = [self.local_AC.h0,self.local_AC.c0]
        
        with tf.GradientTape() as tape:
            policy,_,_,_=self.local_AC(np.stack(rollout[:, 0]),np.stack(rollout[:, 1]),rnn_state)

            optimal_actions_onehot = tf.one_hot(np.stack(rollout[:, 2]), a_size, dtype=tf.float32)

            loss=tf.reduce_mean(tf.keras.backend.categorical_crossentropy(optimal_actions_onehot, policy))

        i_grads = tape.gradient(loss,self.local_AC.trainable_variables)


        return [loss], i_grads

    def calculateGradient(self, rollout, bootstrap_value, episode_count, rnn_state0):
        # ([s,a,r,s1,v[0,0]])

        rollout = np.array(rollout, dtype=object)
        observations = rollout[:, 0]
        goals = rollout[:, -3]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 4]
        valids = rollout[:, 5]
        train_value = rollout[:, -2]
        train_policy = rollout[:, -1]

        #rnn_state = [self.local_AC.h0,self.local_AC.c0]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)


        actions_onehot=tf.one_hot(actions, a_size, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])

        with tf.GradientTape() as tape:
            policy,policy_sig,value,[state_h,state_c]=self.local_AC(np.stack(observations),np.stack(goals),rnn_state0)

            #train_valueはinvalid actionをとったかどうかのラベル
            value_loss=0.1*tf.reduce_mean(train_value*tf.square(np.stack(discounted_rewards)-tf.reshape(value, shape=[-1])))

            entropy     = - tf.reduce_mean(policy * tf.log(tf.clip_by_value(policy, 1e-10, 1.0)))

            policy_loss= - 0.5 * tf.reduce_mean(tf.log(tf.clip_by_value(responsible_outputs, 1e-15, 1.0)) * advantages)

            valid_loss  = - 16 * tf.reduce_mean(tf.log(tf.clip_by_value(policy_sig, 1e-10, 1.0)) * np.stack(valids) + tf.log(tf.clip_by_value(1 - policy_sig, 1e-10, 1.0)) * (1 - np.stack(valids)))


            loss=value_loss+policy_loss+valid_loss-entropy*0.01

        grads=tape.gradient(loss,self.local_AC.trainable_variables)

        var_norms = tf.global_norm(self.local_AC.trainable_variables)
        grads, grad_norms = tf.clip_by_global_norm(grads, GRAD_CLIP)

        return [value_loss, policy_loss, valid_loss, entropy, grad_norms, var_norms], grads

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
            episode_reward = episode_step_count = episode_inv_count = targets_done = episode_stop_count = 0

            # Initial state from the environment
            if self.agentID == 1:
                self.env._reset()
                joint_observations[self.metaAgentID] = self.env._observe()

            self.synchronize()  # synchronize starting time of the threads

            # Get Information For Each Agent 
            validActions = self.env.listValidActions(self.agentID,
                                                        joint_observations[self.metaAgentID][self.agentID])

            s = joint_observations[self.metaAgentID][self.agentID]

            rnn_state = [self.local_AC.h0,self.local_AC.c0]
            rnn_state0 = rnn_state

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

                    a_dist,_,v,rnn_state=self.local_AC(s[0],s[1],rnn_state)

                    skipping_state = False
                    train_policy = train_val = 1

                    if not skipping_state:
                        if not (np.argmax(a_dist.flatten()) in validActions):
                            episode_inv_count += 1
                            train_val = 0
                        train_valid = np.zeros(a_size)
                        train_valid[validActions] = 1

                        valid_dist = np.array([a_dist[0, validActions]])
                        valid_dist /= np.sum(valid_dist)

                        a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                        joint_actions[self.metaAgentID][self.agentID] = a
                        if a == 0:
                            episode_stop_count += 1

                    # Make A Single Agent Gather All Information

                    self.synchronize()

                    if self.agentID == 1:
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
                    r = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID])
                    validActions = self.env.listValidActions(self.agentID, s1)

                    self.synchronize()
                    # Append to Appropriate buffers 
                    if not skipping_state:
                        episode_buffer.append(
                            [s[0], a, joint_rewards[self.metaAgentID][self.agentID], s1, v[0, 0], train_valid, s[1],
                                train_val, train_policy])
                        episode_values.append(v[0, 0])
                    episode_reward += r
                    episode_step_count += 1

                    # Update State
                    s = s1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if (len(episode_buffer) > 1) and (
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

                        else:
                            
                            
                            _,_,s1Value,_=self.local_AC(s[0],s[1],rnn_state)

                        self.loss_metrics, grads = self.calculateGradient(train_buffer, s1Value, episode_count,
                                                                            rnn_state0)

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
