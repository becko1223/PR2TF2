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


class ACNet(tf.keras.Model):
    def __init__(self):
   
        super().__init__()
        w_init = tf.keras.initializers.VarianceScaling()


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

        
                         

        self.policy_layer=layers.Dense(units=A_SIZE,kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))

        self.value=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0))


#inputsで入ってくるのは(step,c,h,w)。最初はstepをバッチであるかのように見せてconvなどの処理をし、その後(step,vector)を(batch,step,vector)にしてlstmに入れる
    def __call__(self,inputs,goal_pos,initial_state):
        x=inputs
        
            
        print(f"#####input shape!!!!!!#####  {x.shape}:{np.array(goal_pos).shape}:{np.array(initial_state).shape}")
        x=tf.transpose(x, perm=[0, 1, 3, 4, 2])

        x=layers.TimeDistributed(self.vgg1_conv1)(x)
        x=layers.TimeDistributed(self.vgg1_conv2)(x)
        x=layers.TimeDistributed(self.vgg1_conv3)(x)
        x=layers.TimeDistributed(self.maxpool1)(x)

        x=layers.TimeDistributed(self.vgg2_conv1)(x)
        x=layers.TimeDistributed(self.vgg2_conv2)(x)
        x=layers.TimeDistributed(self.vgg2_conv3)(x)
        print("xshpae before maxpool2 ",x.shape)
        x=layers.TimeDistributed(self.maxpool2)(x)

        print("xshape before conv3 ",x.shape)
        x=layers.TimeDistributed(self.conv3)(x)
       
        print("x before reshape = ",x.shape)
        x=tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]])
        x=self.actflat(x)

        y=goal_pos
       
        y=self.goal_layer(y)

        print("xshape before concat = ",x.shape)

        x=tf.concat([x,y],-1)

        skip=x

        x=self.h1(x)
        x=self.d1(x)
        x=self.h2(x)
        x=self.d2(x)

        x=self.h3(x+skip)

      

        #x=tf.expand_dims(x,0)
        #x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1], RNN_SIZE])

        print("x shape for lstm = ",x.shape)        
        print("initial_state type for lstm = ",type(initial_state))
        print(type(initial_state[0]), initial_state[0].shape)
        

        lstm_out, state_h, state_c = self.lstm(x, initial_state=initial_state)

        #lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape[2]])
        policy=self.policy_layer(lstm_out)
        policy=tf.nn.softmax(policy)
        policy_sig=tf.sigmoid(policy)

        value=self.value(lstm_out)

        return policy,policy_sig,value,[state_h,state_c]



        



    def _build_net(self, inputs, goal_pos, RNN_SIZE, TRAINING, a_size):
        def conv_mlp(inputs, kernal_size, output_size):
            inputs = tf.reshape(inputs, [-1, 1, kernal_size, 1])
            conv = layers.conv2d(inputs=inputs, padding="VALID", num_outputs=output_size,
                                 kernel_size=[1, kernal_size], stride=1,
                                 data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)

            return conv

        def VGG_Block(inputs):
            def conv_2d(inputs, kernal_size, output_size):
                conv = layers.conv2d(inputs=inputs, padding="SAME", num_outputs=output_size,
                                     kernel_size=[kernal_size[0], kernal_size[1]], stride=1,
                                     data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)

                return conv

            conv1 = conv_2d(inputs, [3, 3], RNN_SIZE // 4)
            conv1a = conv_2d(conv1, [3, 3], RNN_SIZE // 4)
            conv1b = conv_2d(conv1a, [3, 3], RNN_SIZE // 4)
            pool1 = layers.max_pool2d(inputs=conv1b, kernel_size=[2, 2])
            return pool1

        w_init = layers.variance_scaling_initializer()
        vgg1 = VGG_Block(inputs)
        vgg2 = VGG_Block(vgg1)

        conv3 = layers.conv2d(inputs=vgg2, padding="VALID", num_outputs=RNN_SIZE - GOAL_REPR_SIZE, kernel_size=[2, 2],
                              stride=1, data_format="NHWC", weights_initializer=w_init, activation_fn=None)

        flat = tf.nn.relu(layers.flatten(conv3))
        goal_layer = layers.fully_connected(inputs=goal_pos, num_outputs=GOAL_REPR_SIZE)
        hidden_input = tf.concat([flat, goal_layer], 1)
        h1 = layers.fully_connected(inputs=hidden_input, num_outputs=RNN_SIZE)
        d1 = layers.dropout(h1, keep_prob=KEEP_PROB1, is_training=TRAINING)
        h2 = layers.fully_connected(inputs=d1, num_outputs=RNN_SIZE, activation_fn=None)
        d2 = layers.dropout(h2, keep_prob=KEEP_PROB2, is_training=TRAINING)
        self.h3 = tf.nn.relu(d2 + hidden_input)
        # Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(self.h3, [0])
        step_size = tf.shape(inputs)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])

        policy_layer = layers.fully_connected(inputs=self.rnn_out, num_outputs=a_size,
                                              weights_initializer=normalized_columns_initializer(1. / float(a_size)),
                                              biases_initializer=None, activation_fn=None)
        policy = tf.nn.softmax(policy_layer)
        policy_sig = tf.sigmoid(policy_layer)
        value = layers.fully_connected(inputs=self.rnn_out, num_outputs=1,
                                       weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None,
                                       activation_fn=None)

        return policy, value, state_out, state_in, state_init, policy_sig
