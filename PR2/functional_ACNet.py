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


def createmodel():
    w_init = tf.keras.initializers.VarianceScaling()
    ob_inputs=keras.Input(shape=(None,11,11,11))
    goal_inputs=keras.Input(shape=(None,3))

    rnn_state_h_input = keras.Input(shape=(RNN_SIZE,))
    rnn_state_c_input = keras.Input(shape=(RNN_SIZE,))

    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 1, 3, 4, 2]), output_shape=(None, 11,11,11), name='transpose_5d')(ob_inputs)
    x=layers.Reshape((-1,11,11,11))


    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.MaxPool2D(2)(x)

    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.Conv2D(filters=RNN_SIZE // 4,kernel_size=3,strides=1,padding="same",data_format="channels_last",kernel_initializer=w_init, activation='relu')(x)
    x=layers.MaxPool2D(2)(x)

    x=layers.Conv2D(filters=RNN_SIZE - GOAL_REPR_SIZE,kernel_size=2,strides=1,padding="valid",data_format="channels_last",kernel_initializer=w_init, activation=None)(x)

    x = layers.Flatten()(x) 
    x = layers.ReLU()(x)


    y = layers.Reshape((-1, 3), name='reshape_goal_features')(goal_inputs)
    y = layers.Dense(units=GOAL_REPR_SIZE, activation='relu', name='goal_dense')(y)


    x_combined = layers.Concatenate(axis=-1)([x, y])

    skip=x_combined

    x_combined=layers.Dense(units=RNN_SIZE,activation='relu')(x_combined)
    x_combined=layers.Dropout(rate=1-KEEP_PROB1)(x_combined)
    x_combined=layers.Dense(units=RNN_SIZE,activation='relu')(x_combined)
    x_combined=layers.Dropout(rate=1-KEEP_PROB2)(x_combined)

    x_combined = layers.Add()([x_combined, skip])
    x_combined = layers.ReLU()(x_combined)


    def reshape_to_lstm_input(t, obs_shape):
        
        B = tf.shape(obs_shape)[0] 
        S = tf.shape(obs_shape)[1]
        F = RNN_SIZE
        return tf.reshape(t, (B, S, F))
        
    x_lstm = layers.Lambda(lambda t: reshape_to_lstm_input(t, ob_inputs), output_shape=(None, None, RNN_SIZE))(x_combined)

    lstm_out, state_h, state_c=layers.LSTM(units=RNN_SIZE,return_state=True,return_sequences=True)(x_lstm,initial_state=[rnn_state_h_input,rnn_state_c_input])

    policy_layer = layers.Dense(units=A_SIZE, kernel_initializer=NormalizedColumnsInitializer(1.0/float(A_SIZE)))(lstm_out)
    policy = layers.Softmax(name='policy_output')(policy_layer)
    policy_sig = layers.Activation('sigmoid', name='policy_sigmoid')(policy_layer)

    value=layers.Dense(units=1,kernel_initializer=NormalizedColumnsInitializer(1.0))(lstm_out)

    model = keras.Model(
        inputs=[ob_inputs, goal_inputs, rnn_state_h_input, rnn_state_c_input],
        outputs=[policy, policy_sig, value, state_h, state_c]
    )
    
    return model