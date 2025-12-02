import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# parameters for training
GRAD_CLIP = 10.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
RNN_SIZE = 512
FILTER=32
GOAL_REPR_SIZE = 12
A_SIZE=5


# Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class NormalizedColumnsInitializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, shape, dtype=None, **kwargs):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out, dtype=dtype)

    def get_config(self):
        return {'std': self.std}


class ACRDNet(tf.keras.Model):
    def __init__(self):
   
        super().__init__()
        w_init = tf.keras.initializers.VarianceScaling()

        #エンコード
        self.encode_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.encode_res1_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.encode_res1_conv2=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation=None)
        self.encode_layernorm1=layers.LayerNormalization()
        self.encode_res2_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.encode_res2_conv2=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation=None)
        self.encode_layernorm2=layers.LayerNormalization()
        



        #状態遷移
        self.dynamics_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.dynamics_res1_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.dynamics_res1_conv2=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation=None)
        self.dynamics_layernorm1=layers.LayerNormalization()
        self.dynamics_res2_conv1=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.dynamics_res2_conv2=layers.Conv2D(filters=FILTER,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation=None)
        self.dynamics_layernorm2=layers.LayerNormalization()

                         
        #方策、価値、報酬
        self.policy_conv1=layers.Conv2D(filters=FILTER,kernel_size=1,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')  #flattenに際し、チャンネル数を減らす。
        self.policy_layernorm1=layers.LayerNormalization()
        self.policy_flatten=layers.Flatten()
        self.policy_dense1=layers.Dense(units=256,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.policy_layernorm2=layers.LayerNormalization()
        self.policy_dense2=layers.Dense(units=A_SIZE,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))


        self.q1_conv1=layers.Conv2D(filters=FILTER,kernel_size=1,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.q1_layernorm1=layers.LayerNormalization()
        self.q1_flatten=layers.Flatten()
        self.q1_dense1=layers.Dense(units=256,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.q1_layernorm2=layers.LayerNormalization()
        self.q1_dense2=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))


        
        self.q2_conv1=layers.Conv2D(filters=FILTER,kernel_size=1,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.q2_layernorm1=layers.LayerNormalization()
        self.q2_flatten=layers.Flatten()
        self.q2_dense1=layers.Dense(units=256,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.q2_layernorm2=layers.LayerNormalization()
        self.q2_dense2=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))

       
        
        self.reward_conv1=layers.Conv2D(filters=FILTER,kernel_size=1,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.reward_layernorm1=layers.LayerNormalization()
        self.reward_flatten=layers.Flatten()
        self.reward_dense1=layers.Dense(units=256,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.reward_layernorm2=layers.LayerNormalization()
        self.reward_dense2=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))
     



    @tf.function(input_signature=[
                        tf.TensorSpec(shape=[None, None, 11, 11, 11], dtype=tf.float32),  # obs (B, S,C, H, W)
                        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),          # goal (B, S, F)
                    ])
    def encode(self,inputs,goal_pos):
        x=inputs
        
            
        x=tf.transpose(x, perm=[0, 1, 3, 4, 2])

        y=tf.expand_dims(tf.expand_dims(goal_pos,2),2)
        
        y=tf.tile(y,[1,1,11,11,1])

        x=tf.concat([x,y],axis=-1)

        x=layers.TimeDistributed(self.encode_conv1)(x)
        skip=x
        x=layers.TimeDistributed(self.encode_res1_conv1)(x)
        x=layers.TimeDistributed(self.encode_res1_conv2)(x)
        x=x+skip
        x=layers.TimeDistributed(self.encode_layernorm1)(x)

        skip=x
        x=layers.TimeDistributed(self.encode_res2_conv1)(x)
        x=layers.TimeDistributed(self.encode_res2_conv2)(x)
        x=x+skip
        x=layers.TimeDistributed(self.encode_layernorm2)(x)

        
        return x
    

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 11,11,FILTER], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, A_SIZE], dtype=tf.float32)
    ])
    def dynamics(self,latent,action):
        x=latent
        y=action
        y=tf.expand_dims(tf.expand_dims(y,2),2)
        y=tf.tile(y,[1,1,11,11,1])
        x=tf.concat([x,y],axis=-1)
        x=layers.TimeDistributed(self.dynamics_conv1)(x)
        x=layers.TimeDistributed(self.dynamics_res1_conv1)(x)
        x=layers.TimeDistributed(self.dynamics_res1_conv2)(x)
        x=self.dynamics_layernorm1(x)
        x=layers.TimeDistributed(self.dynamics_res2_conv1)(x)
        x=layers.TimeDistributed(self.dynamics_res2_conv2)(x)
        x=self.dynamics_layernorm2(x)
        return x
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None,11, 11, FILTER], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def reward(self,latent,action):
        x=latent
        y=action
        y=tf.expand_dims(tf.expand_dims(y,2),2)
        y=tf.tile(y,[1,1,11,11,1])
        x=tf.concat([x,y],axis=-1)
        x=layers.TimeDistributed(self.reward_conv1)(x)
        x=layers.TimeDistributed(self.reward_layernorm1)(x)
        x = layers.TimeDistributed(self.reward_flatten)(x)
        x=self.reward_dense1(x)
        x=self.reward_layernorm2(x)
        x=self.reward_dense2(x)
        return x
        
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None,11,11, FILTER], dtype=tf.float32)
    ])
    def policy(self,latent):
        x=layers.TimeDistributed(self.policy_conv1)(latent)
        x=layers.TimeDistributed(self.policy_layernorm1)(x)
        x = layers.TimeDistributed(self.policy_flatten)(x)
        x=self.policy_dense1(x)
        x=self.policy_layernorm2(x)
        x=self.policy_dense2(x)
        return x
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None,11,11, FILTER], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def q1(self,latent,action):
        x=latent
        y=action
        y=tf.expand_dims(tf.expand_dims(y,2),2)
        y=tf.tile(y,[1,1,11,11,1])
        x=tf.concat([x,y],axis=-1)
        x=layers.TimeDistributed(self.q1_conv1)(x)
        x=layers.TimeDistributed(self.q1_layernorm1)(x)
        x = layers.TimeDistributed(self.q1_flatten)(x)
        x=self.q1_dense1(x)
        x=self.q1_layernorm2(x)
        x=self.q1_dense2(x)
        return x
  
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None,11,11, FILTER], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def q2(self,latent,action):
        x=latent
        y=action
        y=tf.expand_dims(tf.expand_dims(y,2),2)
        y=tf.tile(y,[1,1,11,11,1])
        x=tf.concat([x,y],axis=-1)
        x=layers.TimeDistributed(self.q2_conv1)(x)
        x=layers.TimeDistributed(self.q2_layernorm1)(x)
        x = layers.TimeDistributed(self.q2_flatten)(x)
        x=self.q2_dense1(x)
        x=self.q2_layernorm2(x)
        x=self.q2_dense2(x)
        return x
   






    
