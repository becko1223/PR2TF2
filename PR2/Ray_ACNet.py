import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# parameters for training
GRAD_CLIP = 10.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
RNN_SIZE = 512
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
        self.vgg1_conv1=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg1_conv2=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg1_conv3=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.maxpool1=layers.MaxPool2D(2)

        self.vgg2_conv1=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg2_conv2=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.vgg2_conv3=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')
        self.maxpool2=layers.MaxPool2D(2)

        self.conv3=layers.Conv2D(filters=RNN_SIZE - GOAL_REPR_SIZE,kernel_size=2,strides=1,padding="valid",data_format="channels_last",kernel_initializer=w_init, activation=None)

        self.flat=layers.Flatten()
        self.actflat=layers.ReLU()

        self.goal_layer=layers.Dense(units=GOAL_REPR_SIZE,activation='relu')


        self.h1=layers.Dense(units=RNN_SIZE,activation='relu')
        self.d1=layers.Dropout(rate=1-KEEP_PROB1)
        self.h2=layers.Dense(units=RNN_SIZE,activation='relu')
        self.d2=layers.Dropout(rate=1-KEEP_PROB2)

        self.h3=layers.ReLU()

        self.lstm=layers.LSTM(units=RNN_SIZE,return_state=True,return_sequences=True)

        self.h0=tf.zeros((1,RNN_SIZE))
        self.c0=tf.zeros((1,RNN_SIZE))


        #状態遷移
        self.dynamics_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.dynamics_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.dynamics_dense3=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="liner")
        

                         
        #方策、価値、報酬
        self.policy_dense1=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.policy_dense2=layers.Dense(units=512,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),activation="elu")
        self.policy_dense3=layers.Dense(units=A_SIZE,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))

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




    def encode(self,inputs,goal_pos,initial_state):
        x=inputs
        
            
        x=tf.transpose(x, perm=[0, 1, 3, 4, 2])

        x=layers.TimeDistributed(self.vgg1_conv1(x))
        x=layers.TimeDistributed(self.vgg1_conv2(x))
        x=layers.TimeDistributed(self.vgg1_conv3(x))
        x=layers.TimeDistributed(self.maxpool1(x))

        x=layers.TimeDistributed(self.vgg2_conv1(x))
        x=layers.TimeDistributed(self.vgg2_conv2(x))
        x=layers.TimeDistributed(self.vgg2_conv3(x))
        x=layers.TimeDistributed(self.maxpool2(x))

        x=layers.TimeDistributed(self.conv3(x))
        x=tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[4]])
        x=self.actflat(x)

        y=goal_pos
        
        y=self.goal_layer(y)

        x=tf.concat([x,y],-1)

        skip=x

        x=self.h1(x)
        x=self.d1(x)
        x=self.h2(x)
        x=self.d2(x)

        x=self.h3(x+skip)


        #x=tf.expand_dims(x,0)
        x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1], RNN_SIZE])
        

        lstm_out, state_h, state_c = self.lstm(x, initial_state=initial_state)
        return lstm_out,[state_h,state_c]
    

    def dynamics(self,latent,action):
        x=tf.concat([latent,action],-1)
        x=self.dynamics_dense1(x)
        x=self.dynamics_dense2(x)
        x=self.dynamics_dense3(x)
        return x
    
    
    def reward(self,latent,action):
        x=tf.concat([latent,action],-1)
        x=self.reward_dense1(x)
        x=self.reward_dense2(x)
        x=self.reward_dense3(x)
        return x
    

    def policy(self,latent):
        x=self.policy_dense1(latent)
        x=self.policy_dense2(x)
        x=self.policy_dense3(x)
      
        return x
    

    def q1(self,latent,action):
        x=tf.concat([latent,action],-1)
        x=self.q1_dense1(x)
        x=self.q1_layernorm(x)
        x=tf.keras.activations.tanh(x)
        x=self.q1_dense2(x)
        x=self.q1_dense3(x)
        return x
    

    def q2(self,latent,action):
        x=tf.concat([latent,action],-1)
        x=self.q2_dense1(x)
        x=self.q2_layernorm(x)
        x=tf.keras.activations.tanh(x)
        x=self.q2_dense2(x)
        x=self.q2_dense3(x)
        return x


#inputsで入ってくるのは(step,c,h,w)。最初はstepをバッチであるかのように見せてconvなどの処理をし、その後(step,vector)を(batch,step,vector)にしてlstmに入れる
    def call(self,inputs,goal_pos,initial_state):
        x=inputs
        if np.array(x).ndim == 3:  
            x = tf.expand_dims(x, axis=0)  
            
        print(f"#####input shape!!!!!!#####  {x.shape}:{np.array(goal_pos).shape}:{np.array(initial_state).shape}")
        x=tf.transpose(x, perm=[0, 2, 3, 1])

        x=self.vgg1_conv1(x)
        x=self.vgg1_conv2(x)
        x=self.vgg1_conv3(x)
        x=self.maxpool1(x)

        x=self.vgg2_conv1(x)
        x=self.vgg2_conv2(x)
        x=self.vgg2_conv3(x)
        x=self.maxpool2(x)

        x=self.conv3(x)
        x=self.flat(x)
        x=self.actflat(x)

        y=goal_pos
        if np.array(y).ndim==1:
            y=tf.expand_dims(y,0)
        y=self.goal_layer(y)

        x=tf.concat([x,y],1)

        skip=x

        x=self.h1(x)
        x=self.d1(x)
        x=self.h2(x)
        x=self.d2(x)

        x=self.h3(x+skip)


        #x=tf.expand_dims(x,0)
        x = tf.reshape(x, [1, -1, RNN_SIZE])
        

        lstm_out, state_h, state_c = self.lstm(x, initial_state=initial_state)

        #lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape[2]])
        policy=self.policy_layer(lstm_out)
        policy=tf.nn.softmax(policy[0])
        policy_sig=tf.sigmoid(policy[0])

        value=self.value(lstm_out)
        reward=self.reward(lstm_out)

        return policy,policy_sig,value,[state_h,state_c],reward



        



    
