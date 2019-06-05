from __future__ import absolute_import, division, print_function, unicode_literals
import setup_path
import airsim
import os
import sys
import time
import numpy as np
import h5py
import argparse
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from coverage_map import CoverageMap
from PIL import Image
import errno

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Desktop\\model270000.ckpt', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\DistributedRL\\Share\\checkpoint\\local_run', type=str)
parser.add_argument('--debug', '-debug', dest='debug', action='store_true')

parser.add_argument('--buffer_size', '-buffer_size', help='number of observations in each sample', default=4, type=int)
parser.add_argument('--input_size', '-input_size', help='width/height of the observation input', default=84, type=int)
parser.add_argument('--cov_method', '-cov_method', help='coverage method to use, choose from [pose, lidar]', default='pose', type=str)

parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--memory_size', '-memory_size', help='number of steps in the replay memory', default=500000, type=int)
parser.add_argument('--train_after', '-train_after', help='number of steps to record before the first training session', default=200000, type=int)
parser.add_argument('--train_interval', '-train_interval', help='number of steps between each training session', default=4, type=int)
parser.add_argument('--target_update_interval', '-target_update_interval', help='number of steps between each target model update', default=10000, type=int)
parser.add_argument('--steps', '-steps', help='number of steps required to lower the epsilon to minimal', default=1000000, type=int)
parser.add_argument('--learning_rate', '-learning_rate', help='learning rate for the optimizer', default=0.00025, type=float)

args = parser.parse_args()

# model definition class
class RLModel(Model):
  def __init__(self, num_actions):
    super(RLModel, self).__init__()
    self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')
    self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')
    self.conv3 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(units=256, activation='relu')
    self.d2 = Dense(units=num_actions, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.
        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.
        Attributes:
            size (int): The minibatch size
        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes
    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.
        Attributes:
            size (int): Minibatch size
        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.
        Attributes:
            index (int): State's index
        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions, metrics_writer, checkpoint_path,
                 gamma=0.99, learning_rate=args.learning_rate, momentum=0.95, minibatch_size=args.batch_size,
                 memory_size=args.memory_size, train_after=args.train_after, train_interval=args.train_interval, target_update_interval=args.target_update_interval, model_path=""):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # initiate model
        self._action_value_net = RLModel(self.nb_actions)

        # load pre trained weights if exist
        if model_path != "":
            previous_steps = int(model_path.split('\\')[-1].split('.')[-2][5:])
            if previous_steps > args.train_after:
                epsilon_steps = args.steps - previous_steps
            else:
                epsilon_steps = args.steps

            self._explorer = LinearEpsilonAnnealingExplorer(1, 0.1, epsilon_steps)
            
            self._action_value_net.load_weights(model_path)
            print('model weights loaded from {}'.format(model_path))

        # initiate loss and optimizer
        self._loss = tf.keras.losses.Huber()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)

        # initiate metrics 
        self._loss_metrics = tf.keras.metrics.Mean(name='train_loss')
        self._loss_summary_writer = metrics_writer

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = RLModel(self.nb_actions)
        self._target_net.set_weights(self._action_value_net.get_weights())
        self._checkpoint_path = checkpoint_path

    def act(self, state, eval=False):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # tf function to predict q values
        @tf.function
        def predict_q_values(image):
            return self._action_value_net(image)

        # Append the state to the short term memory (ie. History)
        self._history.append(state)
        
        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken) and not eval:
            action = self._explorer(self.nb_actions)
            q_values = None
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            env_with_history = np.expand_dims(env_with_history, axis=0)
            env_with_history = env_with_history.transpose(0,2,3,1).astype(np.float64)

            # Append batch axis with only one sample to evaluate
            q_values = predict_q_values(env_with_history).numpy()

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action, q_values

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            #if self._metrics_writer is not None:
            #    self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self, checkpoint_dir):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """
        
        # tf function to train
        @tf.function
        def train_model(pre_states, actions, post_states, rewards, terminals):
            with tf.GradientTape() as tape:

                # Compute the q_targets
                q_targets = tf.where(terminals, rewards, self.gamma * tf.math.reduce_max(self._target_net(post_states), axis=1) + rewards)

                # actions is already a 1-hot encoding of the action done by the agent
                q_acted = tf.math.reduce_sum(self._action_value_net(pre_states) * actions, axis=1)

                # Compute Huber loss on the q values and the predicted ones
                loss = self._loss(q_targets, q_acted)

            gradients = tape.gradient(loss, self._action_value_net.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self._action_value_net.trainable_variables))

            self._loss_metrics(loss)

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                #print('training... number of steps: {}'.format(agent_step))

                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)

                # prepare minibatch for tf training
                pre_states = pre_states.transpose(0, 2, 3, 1).astype(np.float64) # NCHW => NHWC
                post_states = post_states.transpose(0, 2, 3, 1).astype(np.float64) # NCHW => NHWC
                terminals_bool = terminals.astype(bool) # binary => boolean
                rewards = rewards.astype(np.float64)
                actions_onehot = np.zeros((self._minibatch_size, self.nb_actions)) # indices => one hot vectors
                actions_onehot[np.arange(self._minibatch_size), actions] = 1

                train_model(pre_states, actions_onehot, post_states, rewards, terminals_bool)

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    self._target_net.set_weights(self._action_value_net.get_weights())
                    filename = os.path.join(self._checkpoint_path, "model{}.ckpt".format(agent_step))
                    self._action_value_net.save_weights(filename)

                    print('Step: {}, Loss: {}'.format(agent_step, self._loss_metrics.result()))
                    # use tensorboard to inspect loss
                    with self._loss_summary_writer.as_default():
                        tf.summary.scalar('loss', self._loss_metrics.result(), step=agent_step)

    def get_depth_image(self, client):
        # get depth image from airsim
        responses = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])

        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        if img1d.size > 1:

            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((84, 84)).convert('L')) 
            im_final = im_final / 255.0

            return im_final

        return np.zeros((84,84)).astype(float)

    # Gets a coverage image from AirSim
    def get_cov_image(self, coverage_map):

        if args.cov_method == 'pose':
            state, cov_reward = coverage_map.get_state_from_pose()
        else: # args.cov_method = 'lidar'
            state, cov_reward = coverage_map.get_state_from_lidar()

        # debug only
        #im = Image.fromarray(np.uint8(state))
        #im.save("DistributedRL\\debug\\{}.png".format(time.time()))
        
        # normalize state
        state = state / 255.0
        return state, cov_reward

def compute_reward(car_state, cov_reward):

    #print(cov_reward)
    
    alpha = 1.0
    max_speed = 2.5

    if car_state.speed < 0.02:
        return -3.0

    # compute reward based on speed or coverage
    speed_reward = min(1.0, car_state.speed / max_speed)
    reward = alpha * cov_reward + (1 - alpha) * speed_reward
    
    return reward

def isDone(car_state, car_controls, reward, iteration):
    done = 0
    if reward < -1:
        done = 1
    if car_controls.brake == 0:
        if car_state.speed <= 0.02 and iteration < 2:
            done = 1
    return done

def connect_to_airsim():
    attempt_count = 0
    while True:
        try:
            print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count))
            car_client = airsim.CarClient()
            car_client.confirmConnection()
            car_client.enableApiControl(True)
            print('Connected!')
            return car_client
        except:
            print('Failed to connect.')
            attempt_count += 1
            if (attempt_count % 10 == 0):
                print('10 consecutive failures to connect.')

            time.sleep(1)


if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # create rewards txt file
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # make metrics and checkpoints folder
    metrics_dir = os.path.join(args.output_dir,'metrics')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    # metrics file
    metrics_writer = tf.summary.create_file_writer(metrics_dir)

    # connect to airsim
    client = connect_to_airsim()
    car_controls = airsim.CarControls()

    # create coverage map and connect to client
    start_point = [-290.0, 10050.0, 10.0]
    coverage_map = CoverageMap(start_point=start_point, map_size=32000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
    coverage_map.set_client(client)

    # let the car drive a bit
    car_controls.throttle = 0.3
    car_controls.steering = 0
    client.setCarControls(car_controls)
    time.sleep(0.5)

    # Make RL agent
    input_shape = (args.buffer_size, args.input_size, args.input_size)
    actions = [-1.0, -0.5, 0, 0.5, 1.0]
    agent = DeepQAgent(input_shape, len(actions), model_path=args.path, checkpoint_path=checkpoint_dir, metrics_writer=metrics_writer)

    # Train
    current_step = 0
    iterations = 0
    rewards_sum = 0

    # get initial coverage state, after that use the post as the coverage state
    cov_state, cov_reward = agent.get_cov_image(coverage_map)

    while True:

        # get current depth image and combine with coverage state
        state = agent.get_depth_image(client)
        state[:cov_state.shape[0],:cov_state.shape[1]] = cov_state

        # present state if debug mode is on
        if args.debug:
            cv2.imshow('navigation map', state)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # get action and perform
        action, _ = agent.act(state)
        car_controls.steering = actions[action]
        client.setCarControls(car_controls)

        # compute reward and detect terminal state
        car_state = client.getCarState()
        cov_state, cov_reward = agent.get_cov_image(coverage_map)
        #print('reward: {}'.format(cov_reward))
        reward = compute_reward(car_state, cov_reward) 
        done = isDone(car_state, car_controls, reward, iterations)
        if done == 1:
            reward = -10
        
        # sum reward for inspection
        rewards_sum += reward
        iterations += 1

        # train agent
        agent.observe(state, action, reward, done)
        agent.train(args.output_dir)

        if done:
            client.reset()
            car_controls.steering = actions[0]
            client.setCarControls(car_controls)
            time.sleep(1)

            # clear coverage map
            coverage_map.reset()

            # save rewards to view with tensorboard
            with metrics_writer.as_default():
                tf.summary.scalar('rewards sum', rewards_sum, step=current_step)
                tf.summary.scalar('rewards mean', rewards_sum/iterations, step=current_step)

            rewards_sum = 0 # reset reward sum
            iterations = 0

            # let the car drive a bit
            car_controls.throttle = 0.3
            car_controls.steering = 0
            client.setCarControls(car_controls)
            time.sleep(0.5)

            current_step +=1
            #print('number of steps: {}'.format(current_step))
        
        #image = Image.fromarray(np.uint8(current_state))
        #image.save("DistributedRL\\debug\\{}.png".format(time.time()))
