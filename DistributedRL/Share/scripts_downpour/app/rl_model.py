import time
import numpy as np
import json
import threading
import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, clone_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam, Adagrad, Adadelta, RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
from keras.initializers import random_normal

# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch, as it will be unable to 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# A wrapper class for the DQN model
class RlModel():
    def __init__(self, weights_path, train_conv_layers, exp_type='with_rgb', buffer_len=4):

        # actions list. each action will be in the form of [steering, is_reverse]
        self.__actions = [[-1.0, False],
                          [-0.5, False],
                          [0.0, False],
                          [0.5, False],
                          [1.0, False]]

        self.__gamma = 0.99
        self.__exp_type = exp_type

        # Original DQN architecture from "Playing Atari with Deep Reinforcement Learning" 
        # https://arxiv.org/pdf/1312.5602.pdf

        activation = 'relu'

        #pic_input = Input(shape=(59,255,3))
        pic_input = Input(shape=(84,84,buffer_len))
        #if self.__exp_type == 'with_rgb':
        map_input = Input(shape=(40,40))

        # first convolution stack
        img_stack = Conv2D(32, 8, strides=(4, 4), name='convolution_a0', padding='valid', activation=activation, trainable=train_conv_layers)(pic_input)
        img_stack = Conv2D(64, 4, strides=(2, 2), name='convolution_a1', padding='valid', activation=activation, trainable=train_conv_layers)(img_stack)
        img_stack = Conv2D(64, 3, strides=(1, 1), name='convolution_a2', padding='valid', activation=activation, trainable=train_conv_layers)(img_stack)
        #img_stack = Dropout(0.5)(img_stack)
        # flatten
        img_stack = Flatten()(img_stack)

        if self.__exp_type == 'with_rgb':
            # second convolution stack
            #rgb_stack = Conv2D(32, 8, strides=(4, 4), name='convolution_b0', padding='valid', activation=activation, trainable=train_conv_layers)(rgb_input)
            #rgb_stack = Conv2D(64, 4, strides=(2, 2), name='convolution_b1', padding='valid', activation=activation, trainable=train_conv_layers)(rgb_stack)
            #rgb_stack = Conv2D(64, 3, strides=(1, 1), name='convolution_b2', padding='valid', activation=activation, trainable=train_conv_layers)(rgb_stack)
            #rgb_stack = Dropout(0.5)(rgb_stack)
            # flatten
            #rgb_stack = Flatten()(rgb_stack)
            map_stack = Flatten()(map_input)

            # Fully connected layers
            merged_stack = concatenate([img_stack, map_stack])
            #merged_stack = GRU(units=1024, activation=activation)(merged_stack)
            merged_stack = Dense(512, name='rl_dense', activation=activation, kernel_initializer=random_normal(stddev=0.01))(merged_stack)
        else:
            merged_stack = Dense(512, name='rl_dense', activation=activation, kernel_initializer=random_normal(stddev=0.01))(img_stack)

        output = Dense(len(self.__actions), name='rl_output', kernel_initializer=random_normal(stddev=0.01))(merged_stack)

        #opt = Adam()
        opt = RMSprop()

        if self.__exp_type == 'with_rgb':
            self.__action_model = Model(inputs=[pic_input, map_input], outputs=output)
        else:
            self.__action_model = Model(inputs=[pic_input], outputs=output)

        self.__action_model.compile(optimizer=opt, loss='mean_squared_error')
        self.__action_model.summary()
        
        # If we are using pretrained weights for the conv layers, load them and verify the first layer.
        if (weights_path is not None and len(weights_path) > 0):
            print('Loading weights from my_model_weights.h5...')
            print('Current working dir is {0}'.format(os.getcwd()))
            self.__action_model.load_weights(weights_path, by_name=True)
            
            print('First layer: ')
            w = np.array(self.__action_model.get_weights()[0])
            print(w)
        else:
            print('Not loading weights')

        # Set up the target model. 
        # This is a trick that will allow the model to converge more rapidly.
        self.__action_context = tf.get_default_graph()
        self.__target_model = clone_model(self.__action_model)

        self.__target_context = tf.get_default_graph()
        self.__model_lock = threading.Lock()

    # A helper function to read in the model from a JSON packet.
    # This is used both to read the file from disk and from a network packet
    def from_packet(self, packet):
        with self.__action_context.as_default():
            self.__action_model.set_weights([np.array(w) for w in packet['action_model']])
            self.__action_context = tf.get_default_graph()
        if 'target_model' in packet:
            with self.__target_context.as_default():
                self.__target_model.set_weights([np.array(w) for w in packet['target_model']])
                self.__target_context = tf.get_default_graph()

    # A helper function to write the model to a JSON packet.
    # This is used to send the model across the network from the trainer to the agent
    def to_packet(self, get_target = True):
        packet = {}
        with self.__action_context.as_default():
            packet['action_model'] = [w.tolist() for w in self.__action_model.get_weights()]
            self.__action_context = tf.get_default_graph()
        if get_target:
            with self.__target_context.as_default():
                packet['target_model'] = [w.tolist() for w in self.__target_model.get_weights()]

        return packet

    # Updates the model with the supplied gradients
    # This is used by the trainer to accept a training iteration update from the agent
    def update_with_gradient(self, gradients, should_update_critic):
        with self.__action_context.as_default():
            action_weights = self.__action_model.get_weights()
            if (len(action_weights) != len(gradients)):
                raise ValueError('len of action_weights is {0}, but len gradients is {1}'.format(len(action_weights), len(gradients)))
            
            print('UDPATE GRADIENT DEBUG START')
            
            dx = 0
            for i in range(0, len(action_weights), 1):
                action_weights[i] += gradients[i]
                dx += np.sum(np.sum(np.abs(gradients[i])))
            print('Moved weights {0}'.format(dx))
            self.__action_model.set_weights(action_weights)
            self.__action_context = tf.get_default_graph()

            if (should_update_critic):
                with self.__target_context.as_default():
                    print('Updating critic')
                    self.__target_model.set_weights([np.array(w, copy=True) for w in action_weights])
            
            print('UPDATE GRADIENT DEBUG END')
            
    def update_critic(self):
        with self.__target_context.as_default():
            self.__target_model.set_weights([np.array(w, copy=True) for w in self.__action_model.get_weights()])
    
            
    # Given a set of training data, trains the model and determine the gradients.
    # The agent will use this to compute the model updates to send to the trainer
    def get_gradient_update_from_batches(self, batches):
        pre_states = np.array(batches['pre_states'])
        post_states = np.array(batches['post_states'])
        if self.__exp_type == 'with_rgb':
            pre_rgbs = np.array(batches['pre_rgbs'])
            post_rgbs = np.array(batches['post_rgbs'])
        rewards = np.array(batches['rewards'])
        actions = list(batches['actions'])
        is_not_terminal = np.array(batches['is_not_terminal'])

        pre_states = pre_states[:, -1, :, :]
        post_states = post_states[:, -1, :, :]

        pre_rgbs = pre_rgbs.transpose(0, 2, 3, 1)
        post_rgbs = post_rgbs.transpose(0, 2, 3, 1)

        """
        # Our model takes 4 consecutive images as input.
        # NCHW -> NHWC
        
        pre_states = pre_states.transpose(0, 2, 3, 1)
        post_states = post_states.transpose(0, 2, 3, 1)
        
        if self.__exp_type == 'with_rgb':
            # For the next input, only read in the last image from each set of examples
            pre_rgbs = pre_rgbs[:, -1, :, :, :]
            post_rgbs = post_rgbs[:, -1, :, :, :]
        """
        # desired pre states previous shape is: [N, 59, 255, 3]

        print('START GET GRADIENT UPDATE DEBUG')
        
        # We only have labels for the action that the agent actually took.
        # To prevent the model from training the other actions, figure out what the model currently predicts for each input.
        # Then, the gradients with respect to those outputs will always be zero.
        with self.__action_context.as_default():
            if self.__exp_type == 'with_rgb':
                #labels = self.__action_model.predict([pre_states,pre_rgbs], batch_size=32)
                labels = self.__action_model.predict([pre_rgbs, pre_states], batch_size=32)
            else:
                labels = self.__action_model.predict([pre_states], batch_size=32)
        
        # Find out what the target model will predict for each post-decision state.
        with self.__target_context.as_default():
            if self.__exp_type == 'with_rgb':
                #q_futures = self.__target_model.predict([post_states,post_rgbs], batch_size=32)
                q_futures = self.__target_model.predict([post_rgbs, post_states], batch_size=32)
            else:
                q_futures = self.__target_model.predict([post_states], batch_size=32)

        # Apply the Bellman equation
        q_futures_max = np.max(q_futures, axis=1)
        q_labels = (q_futures_max * is_not_terminal * self.__gamma) + rewards
        
        # Update the label only for the actions that were actually taken.
        for i in range(0, len(actions), 1):
            labels[i][actions[i]] = q_labels[i]

        # Perform a training iteration.
        with self.__action_context.as_default():
            original_weights = [np.array(w, copy=True) for w in self.__action_model.get_weights()]
            if self.__exp_type == 'with_rgb':
                #self.__action_model.fit([pre_states,pre_rgbs], labels, epochs=1, batch_size=32, verbose=1)
                self.__action_model.fit([pre_rgbs, pre_states], labels, epochs=1, batch_size=32, verbose=1)
            else:
                self.__action_model.fit([pre_states], labels, epochs=1, batch_size=32, verbose=1)
            
            # Compute the gradients
            new_weights = self.__action_model.get_weights()
            gradients = []
            dx = 0
            for i in range(0, len(original_weights), 1):
                gradients.append(new_weights[i] - original_weights[i])
                dx += np.sum(np.sum(np.abs(new_weights[i]-original_weights[i])))
            print('change in weights from training iteration: {0}'.format(dx))
        
        print('END GET GRADIENT UPDATE DEBUG')

        # Numpy arrays are not JSON serializable by default
        return [w.tolist() for w in gradients]

    # Performs a state prediction given the model input
    def predict_state(self, observation, rgb):
        if (type(observation) == type([])):
            observation = np.array(observation[-1])

        """
            observation = np.array(observation)
        
        if self.__exp_type == 'with_rgb':
            obs_rgb = np.array(rgb[-1])
            obs_rgb = np.expand_dims(obs_rgb, axis=0)

        # Our model takes 4 consecutive images as input.
        observation = observation.transpose(1,2,0)
        observation = np.expand_dims(observation, axis=0)
        """
        observation = np.expand_dims(observation, axis=0)

        obs_rgb = np.array(rgb)
        obs_rgb = obs_rgb.transpose(1,2,0)
        obs_rgb = np.expand_dims(obs_rgb, axis=0)

        with self.__action_context.as_default():
            if self.__exp_type == 'with_rgb':
                #predicted_qs = self.__action_model.predict([observation, obs_rgb])
                predicted_qs = self.__action_model.predict([obs_rgb, observation])
            else:
                predicted_qs = self.__action_model.predict([observation])

        # Select the action with the highest Q value
        predicted_state = np.argmax(predicted_qs)
        return (predicted_state, predicted_qs[0][predicted_state], predicted_qs)

    # Convert the current state to control signals to drive the car.
    # As we are only predicting steering angle, we will use a simple controller to keep the car at a constant speed
    def state_to_control_signals(self, state, car_state):

        if car_state.speed > 4:
            return (self.__actions[state][0], self.__actions[state][1], 1.0)
        else:
            return (self.__actions[state][0], self.__actions[state][1], 0.0)

    # Gets a random state
    # Used during annealing
    def get_random_state(self):
        return np.random.randint(low=0, high=len(self.__actions))
