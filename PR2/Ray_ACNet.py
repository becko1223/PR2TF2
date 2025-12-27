import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# parameters for training
GRAD_CLIP = 10.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
ENCODE_SIZE = 512
FILTER=32
GOAL_REPR_SIZE = 12
A_SIZE=5
from parameters import horizon


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


class ResBlock(layers.Layer):
    """Residual Block for keeping spatial info without pooling"""
    def __init__(self, filters, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same", activation="elu")
        self.norm1 = layers.LayerNormalization(axis=-1) # Channel-wise LN
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same", activation=None)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.act = layers.Activation("elu")

        # チャンネル数が変わる場合の調整用
        self.residual_conv = layers.Conv2D(filters, 1, padding="same")

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        if x.shape[-1] != residual.shape[-1]:
             residual = self.residual_conv(residual)
             
        return self.act(x + residual)

class MLPBlock(layers.Layer):
    """MLP Block with LayerNorm and Residual Connection"""
    def __init__(self, units):
        super(MLPBlock, self).__init__()
        self.dense1 = layers.Dense(units, activation="elu")
        self.norm1 = layers.LayerNormalization()
        self.dense2 = layers.Dense(units, activation=None)
        self.norm2 = layers.LayerNormalization()
        self.act = layers.Activation("elu")

    def call(self, x):
        residual = x
        x = self.dense1(x)
        x = self.norm1(x)
        x = self.dense2(x)
        x = self.norm2(x)
        return self.act(x + residual)


class ACRDNet(tf.keras.Model):
    def __init__(self):
   
        super().__init__()
        w_init = tf.keras.initializers.VarianceScaling()

        """
        #エンコード
        self.vgg1_conv1=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg1_conv2=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg1_conv3=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.maxpool1=layers.MaxPool2D(2)

        self.vgg2_conv1=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg2_conv2=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg2_conv3=layers.Conv2D(filters=ENCODE_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.maxpool2=layers.MaxPool2D(2)

        self.conv3=layers.Conv2D(filters=ENCODE_SIZE - GOAL_REPR_SIZE,kernel_size=2,strides=1,padding="valid",data_format="channels_last",kernel_initializer=w_init, activation=None)

        self.flat=layers.Flatten()
        self.actflat=layers.ReLU()

        self.goal_layer=layers.Dense(units=GOAL_REPR_SIZE,activation='relu')


        self.h1=layers.Dense(units=ENCODE_SIZE,activation='relu')
        self.d1=layers.Dropout(rate=1-KEEP_PROB1)
        self.h2=layers.Dense(units=ENCODE_SIZE,activation='relu')
        self.d2=layers.Dropout(rate=1-KEEP_PROB2)

        self.h3=layers.ReLU()

        self.lstm=layers.LSTM(units=ENCODE_SIZE,return_state=True,return_sequences=True)

        
        #コミュニケーション
        self.mha=layers.MultiHeadAttention(num_heads=8,key_dim=64,dropout=0.1)
        self.mha_layernorm=layers.LayerNormalization()
        self.feedforward1=layers.Dense(units=2048,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="relu")
        self.feedforward2=layers.Dense(units=ENCODE_SIZE+(horizon-1)*A_SIZE,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="linear")
        self.ff_layernorm=layers.LayerNormalization()
        self.mha_last_dense=layers.Dense(units=ENCODE_SIZE,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="linear")


        #状態遷移
        self.dynamics_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.dynamics_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.dynamics_dense3=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="linear")
        

                         
        #方策、価値、報酬
        self.policy_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.policy_dense2=layers.Dense(units=A_SIZE,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))


        self.q1_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None))
        self.q1_layernorm=layers.LayerNormalization()
        #実行時tanh
        self.q1_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.q1_dense3=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0))



        self.q2_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None))
        self.q2_layernorm=layers.LayerNormalization()
        #実行時tanh
        self.q2_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.q2_dense3=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0))
       
        
        self.reward_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.reward_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.reward_dense3=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0))
        """


        self.conv_entry = layers.Conv2D(32, 3, padding="same", activation="elu")
        self.res_block1 = ResBlock(32)
        self.res_block2 = ResBlock(64) 
        self.res_block3 = ResBlock(64)
        
        self.flat = layers.Flatten()
        self.goal_layer = layers.Dense(units=GOAL_REPR_SIZE, activation="elu")
        
        # Encoder Projection
        self.pre_dense = layers.Dense(ENCODE_SIZE, activation="elu")
        self.encode_res = MLPBlock(512)
        self.encode_norm = layers.LayerNormalization()

        # --- Communication (Multi-Head Attention) ---
        self.mha = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0) # DropoutはRLでは0が良いことが多い
        self.mha_ln1 = layers.LayerNormalization()
        self.mha_ff = layers.Dense(ENCODE_SIZE+(horizon-1)*A_SIZE, activation="elu")
        self.mha_ln2 = layers.LayerNormalization()
        # プロジェクション層を追加して次元を合わせる
        self.comm_out = layers.Dense(ENCODE_SIZE, activation=None)

        # --- Dynamics (Residual MLP) ---
        # 入力を潜在空間に変換する層
        self.dyn_embed = layers.Dense(512, activation="elu")
        self.dyn_res1 = MLPBlock(512)
        self.dyn_res2 = MLPBlock(512)
        # 次の状態への変化量(delta)を出力すると学習しやすい
        self.dyn_out = layers.Dense(ENCODE_SIZE, activation=None) 
        self.dyn_norm = layers.LayerNormalization()

        # --- Reward ---
        self.rew_embed = layers.Dense(256, activation="elu")
        self.rew_res1 = MLPBlock(256)
        self.rew_out = layers.Dense(1, activation=None)

        # --- Policy ---
        self.pi_embed = layers.Dense(256, activation="elu")
        self.pi_res1 = MLPBlock(256)
        self.pi_out = layers.Dense(A_SIZE, activation=None) # Logits

        # --- Value (Q1 / Q2) ---
        self.q1_embed = layers.Dense(256, activation="elu")
        self.q1_res1 = MLPBlock(256)
        self.q1_out = layers.Dense(1, activation=None)

        self.q2_embed = layers.Dense(256, activation="elu")
        self.q2_res1 = MLPBlock(256)
        self.q2_out = layers.Dense(1, activation=None)



    @tf.function(input_signature=[
                        tf.TensorSpec(shape=[None, None, 4, 11, 11], dtype=tf.float32),  # obs (B, S,C, H, W)
                        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),          # goal (B, S, F)
                    ])
    def encode(self,inputs,goal_pos):
        x=inputs
        
            
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        
        # TimeDistributedでバッチ×時間をまとめて処理
        x = layers.TimeDistributed(self.conv_entry)(x)
        x = layers.TimeDistributed(self.res_block1)(x)
        x = layers.TimeDistributed(self.res_block2)(x)
        x = layers.TimeDistributed(self.res_block3)(x) # 11x11x64
        
        x = layers.TimeDistributed(self.flat)(x) # Flatten
        
        # Goal processing
        g = self.goal_layer(goal_pos)
        
        # Merge
        x = tf.concat([x, g], axis=-1)
        x = self.pre_dense(x)
  
        x = self.encode_res(x)
        x = self.encode_norm(x)
        return x
   
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 1, ENCODE_SIZE+(horizon-1)*A_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, ENCODE_SIZE+(horizon-1)*A_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1, None], dtype=tf.bool)
    ])
    def communication(self,own_vec,all_vec,mask):
        B = tf.shape(own_vec)[0]
        T = tf.shape(own_vec)[1]
        D = tf.shape(own_vec)[3] # 特徴量次元
        L = tf.shape(all_vec)[2] # 周囲のエージェント数
        
        own_flat = tf.reshape(own_vec, [B*T, 1, D])
        all_flat = tf.reshape(all_vec, [B*T, L, D])
        mask_flat = tf.reshape(mask, [B*T, 1, L])
        
        # Attention
        attn_out = self.mha(query=own_flat, value=all_flat, key=all_flat, attention_mask=mask_flat)
        
        # Add & Norm (Residual connection)
        x = self.mha_ln1(own_flat + attn_out)
        
        # Feed Forward
        ff = self.mha_ff(x)
        x = self.mha_ln2(x + ff)
        
        x = self.comm_out(x)
        return tf.reshape(x, [B, T, ENCODE_SIZE])



    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, ENCODE_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, A_SIZE], dtype=tf.float32)
    ])
    def dynamics(self,latent,action):
        inp = tf.concat([latent, action], axis=-1)
        x = self.dyn_embed(inp)
        x = self.dyn_res1(x)
        x = self.dyn_res2(x)
        delta = self.dyn_out(x)
        
        # Residual Dynamics: 次の状態 = 現在の状態 + 変化量
        x = latent + delta
        x = self.dyn_norm(x)
        return x
        
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, ENCODE_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def reward(self,latent,action):
        inp = tf.concat([latent, action], axis=-1)
        x = self.rew_embed(inp)
        x = self.rew_res1(x)
        return self.rew_out(x)
        
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, ENCODE_SIZE], dtype=tf.float32)
    ])
    def policy(self,latent):
        x = self.pi_embed(latent)
        x = self.pi_res1(x)
        return self.pi_out(x)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, ENCODE_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def q1(self,latent,action):
        inp = tf.concat([latent, action], axis=-1)
        x = self.q1_embed(inp)
        x = self.q1_res1(x)
        return self.q1_out(x)
  
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, ENCODE_SIZE], dtype=tf.float32),
        tf.TensorSpec(shape=[None,None,A_SIZE],dtype=tf.float32)
    ])
    def q2(self,latent,action):
        inp = tf.concat([latent, action], axis=-1)
        x = self.q2_embed(inp)
        x = self.q2_res1(x)
        return self.q2_out(x)
   






    
