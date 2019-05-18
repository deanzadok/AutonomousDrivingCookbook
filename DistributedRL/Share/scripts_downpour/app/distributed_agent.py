#from airsim_client import *
from rl_model import RlModel
import setup_path 
import airsim
import msgpackrpc
import time
import math
import numpy as np
import threading
import json
import os
import uuid
import glob
import datetime
import h5py
import sys
import requests
import PIL
import copy
import datetime
from coverage_map import CoverageMap
import errno
from random import randint

# A class that represents the agent that will drive the vehicle, train the model, and send the gradient updates to the trainer.
class DistributedAgent():
    def __init__(self, parameters):
        required_parameters = ['data_dir', 'max_epoch_runtime_sec', 'replay_memory_size', 'batch_size', 'min_epsilon', 'per_iter_epsilon_reduction', 'experiment_name', 'train_conv_layers', 'exp_type', 'start_x', 'start_y', 'start_z', 'log_path']
        for required_parameter in required_parameters:
            if required_parameter not in parameters:
                raise ValueError('Missing required parameter {0}'.format(required_parameter))

        parameters['role_type'] = 'agent'

        
        print('Starting time: {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)
        self.__model_buffer = None
        self.__model = None
        self.__airsim_started = False
        self.__data_dir = parameters['data_dir']
        self.__per_iter_epsilon_reduction = float(parameters['per_iter_epsilon_reduction'])
        self.__min_epsilon = float(parameters['min_epsilon'])
        self.__max_epoch_runtime_sec = float(parameters['max_epoch_runtime_sec'])
        self.__replay_memory_size = int(parameters['replay_memory_size'])
        self.__batch_size = int(parameters['batch_size'])
        self.__experiment_name = parameters['experiment_name']
        self.__train_conv_layers = bool((parameters['train_conv_layers'].lower().strip() == 'true'))
        self.__epsilon = 1
        self.__num_batches_run = 0
        self.__last_checkpoint_batch_count = 0
        
        if 'batch_update_frequency' in parameters:
            self.__batch_update_frequency = int(parameters['batch_update_frequency'])
        
        if 'weights_path' in parameters:
            self.__weights_path = parameters['weights_path']
        else:
            self.__weights_path = None
            
        if 'airsim_path' in parameters:
            self.__airsim_path = parameters['airsim_path']
        else:
            self.__airsim_path = None

        self.__local_run = 'local_run' in parameters

        self.__car_client = None
        self.__car_controls = None

        self.__minibatch_dir = os.path.join(self.__data_dir, 'minibatches')
        self.__output_model_dir = os.path.join(self.__data_dir, 'models')

        self.__make_dir_if_not_exist(self.__minibatch_dir)
        self.__make_dir_if_not_exist(self.__output_model_dir)
        self.__last_model_file = ''

        self.__possible_ip_addresses = []
        self.__trainer_ip_address = None

        self.__experiences = {}

        self.__start_point = [float(parameters['start_x']), float(parameters['start_y']), float(parameters['start_z'])]
        self.__log_file = parameters['log_path']
        self.__exp_type = parameters['exp_type']

        # initiate coverage map
        self.__coverage_map = CoverageMap(start_point=self.__start_point, map_size=12000, scale_ratio=20, state_size=4000, input_size=40, height_threshold=0.9, reward_norm=2.0)

        # create txt file
        if not os.path.isdir(os.path.join(self.__data_dir,'\\checkpoint',self.__experiment_name)):
            os.makedirs(os.path.join(self.__data_dir,'\\checkpoint',self.__experiment_name))
        self.__rewards_log = open(os.path.join(self.__data_dir,'\\checkpoint',self.__experiment_name,"rewards.txt"),"w")
        self.__rewards_log.write("Timestamp\tSum\tMean\n")
        self.__rewards_log.close()

        # create starting points list
        self.__starting_points = self.__get_starting_points()
        #self.__starting_points = [[500.0, 850.0, 32.0]]
    # Starts the agent
    def start(self):
        self.__run_function()

    # The function that will be run during training.
    # It will initialize the connection to the trainer, start AirSim, and continuously run training iterations.
    def __run_function(self):
        print('Starting run function')
        
        # Once the trainer is online, it will write its IP to a file in (data_dir)\trainer_ip\trainer_ip.txt
        # Wait for that file to exist
        if not self.__local_run:
            print('Waiting for trainer to come online')
            while True:
                trainer_ip_dir = os.path.join(os.path.join(self.__data_dir, 'trainer_ip'), self.__experiment_name)
                print('Checking {0}...'.format(trainer_ip_dir))
                if os.path.isdir(trainer_ip_dir):
                    with open(os.path.join(trainer_ip_dir, 'trainer_ip.txt'), 'r') as f:
                        self.__possible_ip_addresses.append(f.read().replace('\n', ''))
                        break
                print('Not online yet. Sleeping...')
                time.sleep(5)
        
            # We now have the IP address for the trainer. Attempt to ping the trainer.
            ping_idx = -1
            while True:
                ping_idx += 1
                print('Attempting to ping trainer...')
                try:
                    print('\tPinging {0}...'.format(self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)]))
                    response = requests.get('http://{0}:80/ping'.format(self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)])).json()
                    if response['message'] != 'pong':
                        raise ValueError('Received unexpected message: {0}'.format(response))
                    print('Success!')
                    self.__trainer_ip_address = self.__possible_ip_addresses[ping_idx % len(self.__possible_ip_addresses)]
                    break
                except Exception as e:
                    print('Could not get response. Message is {0}'.format(e))
                    if (ping_idx % len(self.__possible_ip_addresses) == 0):
                        print('Waiting 5 seconds and trying again...')
                        time.sleep(5)

            # Get the latest model from the trainer
            print('Getting model from the trainer')
            sys.stdout.flush()
            buffer_len = 4
            self.__model = RlModel(weights_path=self.__weights_path, train_conv_layers=self.__train_conv_layers, exp_type=self.__exp_type, buffer_len=buffer_len)
            self.__get_latest_model()
        
        else:
            print('Run is local. Skipping connection to trainer.')
            buffer_len = 4
            self.__model = RlModel(weights_path=self.__weights_path, train_conv_layers=self.__train_conv_layers, exp_type=self.__exp_type, buffer_len=buffer_len)
            
        # Connect to the AirSim exe
        self.__connect_to_airsim()

        # Fill the replay memory by driving randomly.
        print('Filling replay memory...')
        while True:
            print('Running Airsim Epoch.')
            try:
                _, num_of_actions = self.__run_airsim_epoch(True)
                if num_of_actions > 0:
                    percent_full = 100.0 * len(self.__experiences['actions'])/self.__replay_memory_size
                    print('Replay memory now contains {0} members. ({1}% full)'.format(len(self.__experiences['actions']), percent_full))

                    if (percent_full >= 100.0):
                        break
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim while filling replay memory. Attempting to reconnect.')
                self.__connect_to_airsim()
            
        # Get the latest model. Other agents may have finished before us.
        print('Replay memory filled. Starting main loop...')
        
        if not self.__local_run:
            self.__get_latest_model()
        while True:
            try:
                if (self.__model is not None):

                    #Generate a series of training examples by driving the vehicle in AirSim
                    print('Running Airsim Epoch.')
                    experiences, frame_count = self.__run_airsim_epoch(False)

                    # If we didn't immediately crash, train on the gathered experiences
                    if (frame_count > 0):
                        print('Generating {0} minibatches...'.format(frame_count))

                        print('Sampling Experiences.')
                        # Sample experiences from the replay memory
                        sampled_experiences = self.__sample_experiences(experiences, frame_count, True)

                        self.__num_batches_run += frame_count
                        
                        # If we successfully sampled, train on the collected minibatches and send the gradients to the trainer node
                        if (len(sampled_experiences) > 0):
                            print('Publishing AirSim Epoch.')

                            # write all rewards to log file
                            self.__rewards_log = open(os.path.join(self.__data_dir,'checkpoint',self.__experiment_name,"rewards.txt"),"a+")
                            rewards_sum = 0
                            for reward in sampled_experiences['rewards']:
                                rewards_sum += reward
                            self.__rewards_log.write("{}\t{}\t{}\n".format(time.time(),rewards_sum,rewards_sum/len(sampled_experiences['rewards'])))
                            self.__rewards_log.close()

                            self.__publish_batch_and_update_model(sampled_experiences, frame_count)
            
            # Occasionally, the AirSim exe will stop working.
            # For example, if a user connects to the node to visualize progress.
            # In that case, attempt to reconnect.
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim. Attempting to reconnect.')
                self.__connect_to_airsim()

    # Connects to the AirSim Exe.
    # Assume that it is already running. After 10 successive attempts, attempt to restart the executable.
    def __connect_to_airsim(self):
        attempt_count = 0
        while True:
            try:
                print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count))
                self.__car_client = airsim.CarClient()
                self.__car_client.confirmConnection()
                self.__car_client.enableApiControl(True)
                self.__car_controls = airsim.CarControls()
                self.__coverage_map.set_client(client=self.__car_client) # update client on coverage map
                print('Connected!')
                return
            except:
                print('Failed to connect.')
                attempt_count += 1
                if (attempt_count % 10 == 0):
                    print('10 consecutive failures to connect. Attempting to start AirSim on my own.')
                    
                    if self.__local_run:
                        os.system('START "" powershell.exe {0}'.format(os.path.join(self.__airsim_path, 'AD_Cookbook_Start_AirSim.ps1 neighborhood -windowed')))
                    else:
                        os.system('START "" powershell.exe D:\\AD_Cookbook_AirSim\\Scripts\\DistributedRL\\restart_airsim_if_agent.ps1')
                print('Waiting a few seconds.')
                time.sleep(10)

    # Appends a sample to a ring buffer.
    # If the appended example takes the size of the buffer over buffer_size, the example at the front will be removed.
    def __append_to_ring_buffer(self, item, rgb_item, buffer, rgb_buffer, buffer_size):
        if (len(buffer) >= buffer_size):
            buffer = buffer[1:]
            rgb_buffer = rgb_buffer[1:]
        buffer.append(item)
        rgb_buffer.append(rgb_item)
        return buffer, rgb_buffer
    
    # Runs an interation of data generation from AirSim.
    # Data will be saved in the replay memory.
    def __run_airsim_epoch(self, always_random):
        print('Running AirSim epoch.')
        
        # reset coverage map
        self.__coverage_map.reset()

        # Pick a random starting point on the roads
        starting_points, starting_direction = self.__get_next_starting_point()
        
        # Initialize the state buffer.
        # For now, save 4 images at 0.01 second intervals.
        state_buffer_len = 4
        state_buffer = []
        rgb_buffer = []
        wait_delta_sec = 0.025

        print('Getting Pose')
        self.__car_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(starting_points[0], starting_points[1], starting_points[2]), toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        # Currently, simSetVehiclePose does not allow us to set the velocity. 
        # So, if we crash and call simSetVehiclePose, the car will be still moving at its previous velocity.
        # We need the car to stop moving, so push the brake and wait for a few seconds.
        print('Waiting for momentum to die')
        self.__car_controls.steering = 0
        self.__car_controls.throttle = 0
        self.__car_controls.brake = 1
        self.__car_client.setCarControls(self.__car_controls)
        time.sleep(4)
        
        print('Resetting')
        self.__car_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(starting_points[0], starting_points[1], starting_points[2]), toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        #Start the car rolling so it doesn't get stuck
        print('Running car for a few seconds...')
        self.__car_controls.steering = 0
        self.__car_controls.throttle = 0.4
        self.__car_controls.brake = 0
        self.__car_client.setCarControls(self.__car_controls)
        
        # While the car is rolling, start initializing the state buffer
        stop_run_time =datetime.datetime.now() + datetime.timedelta(seconds=2)
        while(datetime.datetime.now() < stop_run_time):
            time.sleep(wait_delta_sec)
            cov_image, _ = self.__get_cov_image()
            rgb_image = None
            if self.__exp_type == 'with_rgb':
                rgb_image = self.__get_image()
            state_buffer, rgb_buffer = self.__append_to_ring_buffer(cov_image, rgb_image, state_buffer, rgb_buffer, state_buffer_len)
        done = False
        actions = [] #records the state we go to
        pre_states = []
        post_states = []
        if self.__exp_type == 'with_rgb':
            pre_rgbs = []
            post_rgbs = []
        rewards = []
        predicted_rewards = []
        car_state = self.__car_client.getCarState()

        # slow down a bit
        self.__car_controls.throttle = 0.3
        self.__car_client.setCarControls(self.__car_controls)

        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_epoch_runtime_sec)
        
        num_random = 0
        
        # Main data collection loop
        while not done:
            collision_info = self.__car_client.simGetCollisionInfo()
            utc_now = datetime.datetime.utcnow()
            
            # Check for terminal conditions:
            # 1) Car has collided
            # 2) Car is stopped
            # 3) The run has been running for longer than max_epoch_runtime_sec. 
            #       This constraint is so the model doesn't end up having to churn through huge chunks of data, slowing down training
            if (collision_info.has_collided or abs(car_state.speed) < 0.02 or utc_now > end_time):
                print('Start time: {0}, end time: {1}'.format(start_time, utc_now), file=sys.stderr)
                if (utc_now > end_time):
                    print('timed out.')
                    print('Full autonomous run finished at {0}'.format(utc_now), file=sys.stderr)
                done = True
                sys.stderr.flush()
            else:

                # The Agent should occasionally pick random action instead of best action
                do_greedy = np.random.random_sample()
                pre_state = copy.deepcopy(state_buffer)
                if self.__exp_type == 'with_rgb':
                    pre_rgb = copy.deepcopy(rgb_buffer)
                else:
                    pre_rgb = None
                if (do_greedy < self.__epsilon or always_random):
                    num_random += 1
                    next_state = self.__model.get_random_state()
                    predicted_reward = 0
                    
                else:
                    next_state, predicted_reward, _ = self.__model.predict_state(pre_state,pre_rgb)

                    print('Model predicts {0}'.format(next_state))

                # Convert the selected state to a control signal
                next_steering, is_reverse, next_brake = self.__model.state_to_control_signals(next_state, self.__car_client.getCarState())

                # penalty acquired from changing driving direction
                drive_change_penalty = False 

                # Take the action
                self.__car_controls.steering = next_steering
                self.__car_controls.brake = next_brake
                if is_reverse:
                    if self.__car_controls.throttle == 0.3:
                        drive_change_penalty = True
                    self.__car_controls.throttle = -0.3
                    self.__car_controls.is_manual_gear = True
                    self.__car_controls.manual_gear = -1
                else:
                    if self.__car_controls.throttle == -0.3:
                        drive_change_penalty = True
                    self.__car_controls.throttle = 0.3
                    self.__car_controls.is_manual_gear = False
                    self.__car_controls.manual_gear = 0

                self.__car_client.setCarControls(self.__car_controls)
                
                # Wait for a short period of time to see outcome
                time.sleep(wait_delta_sec)

                # Observe outcome and compute reward from action
                post_cov_image, cov_reward = self.__get_cov_image()
                if self.__exp_type == 'with_rgb':
                    post_rgb_image = self.__get_image()
                else:
                    post_rgb_image = None

                state_buffer, rgb_buffer = self.__append_to_ring_buffer(post_cov_image, post_rgb_image, state_buffer, rgb_buffer, state_buffer_len)
                car_state = self.__car_client.getCarState()
                collision_info = self.__car_client.simGetCollisionInfo()
                reward = self.__compute_reward(collision_info, car_state, cov_reward, next_state)
                
                # Add the experience to the set of examples from this iteration
                pre_states.append(pre_state)
                post_states.append(state_buffer)
                if self.__exp_type == 'with_rgb':
                    pre_rgbs.append(pre_rgb)
                    post_rgbs.append(rgb_buffer)
                rewards.append(reward)
                predicted_rewards.append(predicted_reward)
                actions.append(next_state)

        # Only the last state is a terminal state.
        is_not_terminal = [1 for i in range(0, len(actions)-1, 1)]
        is_not_terminal.append(0)
        
        # only add to the replay memory if have enough data
        if len(actions) > 30:
            # Add all of the states from this iteration to the replay memory
            self.__add_to_replay_memory('pre_states', pre_states)
            self.__add_to_replay_memory('post_states', post_states)
            if self.__exp_type == 'with_rgb':
                self.__add_to_replay_memory('pre_rgbs', pre_rgbs)
                self.__add_to_replay_memory('post_rgbs', post_rgbs)
            self.__add_to_replay_memory('actions', actions)
            self.__add_to_replay_memory('rewards', rewards)
            self.__add_to_replay_memory('predicted_rewards', predicted_rewards)
            self.__add_to_replay_memory('is_not_terminal', is_not_terminal)

            print('Percent random actions: {0}'.format(num_random / max(1, len(actions))))
            print('Num total actions: {0}'.format(len(actions)))
            
            # If we are in the main loop, reduce the epsilon parameter so that the model will be called more often
            # Note: this will be overwritten by the trainer's epsilon if running in distributed mode
            if not always_random:
                self.__epsilon -= self.__per_iter_epsilon_reduction
                self.__epsilon = max(self.__epsilon, self.__min_epsilon)
            
            return self.__experiences, len(actions)
        else:
            return self.__experiences, 0
            

    # Adds a set of examples to the replay memory
    def __add_to_replay_memory(self, field_name, data):
        if field_name not in self.__experiences:
            self.__experiences[field_name] = data
        else:
            self.__experiences[field_name] += data
            start_index = max(0, len(self.__experiences[field_name]) - self.__replay_memory_size)
            self.__experiences[field_name] = self.__experiences[field_name][start_index:]

    # Sample experiences from the replay memory
    def __sample_experiences(self, experiences, frame_count, sample_randomly):
        sampled_experiences = {}
        sampled_experiences['pre_states'] = []
        sampled_experiences['post_states'] = []
        if self.__exp_type == 'with_rgb':
            sampled_experiences['pre_rgbs'] = []
            sampled_experiences['post_rgbs'] = []
        sampled_experiences['actions'] = []
        sampled_experiences['rewards'] = []
        sampled_experiences['predicted_rewards'] = []
        sampled_experiences['is_not_terminal'] = []

        # Compute the surprise factor, which is the difference between the predicted an the actual Q value for each state.
        # We can use that to weight examples so that we are more likely to train on examples that the model got wrong.
        suprise_factor = np.abs(np.array(experiences['rewards'], dtype=np.dtype(float)) - np.array(experiences['predicted_rewards'], dtype=np.dtype(float)))
        suprise_factor_normalizer = np.sum(suprise_factor)
        suprise_factor /= float(suprise_factor_normalizer)

        # Generate one minibatch for each frame of the run
        for _ in range(0, frame_count, 1):
            if sample_randomly:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False))
            else:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False, p=suprise_factor))
        
            sampled_experiences['pre_states'] += [experiences['pre_states'][i] for i in idx_set]
            sampled_experiences['post_states'] += [experiences['post_states'][i] for i in idx_set]
            if self.__exp_type == 'with_rgb':
                sampled_experiences['pre_rgbs'] += [experiences['pre_rgbs'][i] for i in idx_set]
                sampled_experiences['post_rgbs'] += [experiences['post_rgbs'][i] for i in idx_set]
            sampled_experiences['actions'] += [experiences['actions'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            
        return sampled_experiences
        
     
    # Train the model on minibatches and post to the trainer node.
    # The trainer node will respond with the latest version of the model that will be used in further data generation iterations.
    def __publish_batch_and_update_model(self, batches, batches_count):
        # Train and get the gradients
        print('Publishing epoch data and getting latest model from parameter server...')
        gradients = self.__model.get_gradient_update_from_batches(batches)
        
        # Post the data to the trainer node
        if not self.__local_run:
            post_data = {}
            post_data['gradients'] = gradients
            post_data['batch_count'] = batches_count
            
            response = requests.post('http://{0}:80/gradient_update'.format(self.__trainer_ip_address), json=post_data)
            print('Response:')
            print(response)

            new_model_parameters = response.json()
            
            # Update the existing model with the new parameters
            self.__model.from_packet(new_model_parameters)
            
            #If the trainer sends us a epsilon, allow it to override our local value
            if ('epsilon' in new_model_parameters):
                new_epsilon = float(new_model_parameters['epsilon'])
                print('Overriding local epsilon with {0}, which was sent from trainer'.format(new_epsilon))
                self.__epsilon = new_epsilon
                
        else:
            if (self.__num_batches_run > self.__batch_update_frequency + self.__last_checkpoint_batch_count):
                self.__model.update_critic()
                
                checkpoint = {}
                checkpoint['model'] = self.__model.to_packet(get_target=True)
                checkpoint['batch_count'] = batches_count
                checkpoint_str = json.dumps(checkpoint)

                checkpoint_dir = os.path.join(os.path.join(self.__data_dir, 'checkpoint'), self.__experiment_name)
                
                if not os.path.isdir(checkpoint_dir):
                    try:
                        os.makedirs(checkpoint_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                            
                file_name = os.path.join(checkpoint_dir,'{0}.json'.format(self.__num_batches_run)) 
                with open(file_name, 'w') as f:
                    print('Checkpointing to {0}'.format(file_name))
                    f.write(checkpoint_str)
                
                self.__last_checkpoint_batch_count = self.__num_batches_run
                
    # Gets the latest model from the trainer node
    def __get_latest_model(self):
        print('Getting latest model from parameter server...')
        response = requests.get('http://{0}:80/latest'.format(self.__trainer_ip_address)).json()
        self.__model.from_packet(response)

    # Gets a coverage image from AirSim
    def __get_cov_image(self):

        state, cov_reward = self.__coverage_map.get_state()
        state = self.__coverage_map.get_map_scaled()

        # debug only
        #im = PIL.Image.fromarray(np.uint8(state))
        #im.save("DistributedRL\\debug\\{}.png".format(time.time()))
        
        # normalize state
        state = state / 255.0

        return state, cov_reward

    # Gets an image from AirSim
    def __get_image(self):

        responses = self.__car_client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
        return transform_depth_input(responses)
        """
        image_response = self.__car_client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)])[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        if image1d.size > 1:
            image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

            #im = PIL.Image.fromarray(np.uint8(image_rgba))
            #im.save("DistributedRL\\debug\\{}.png".format(time.time()))

            image_rgba = image_rgba / 255.0

            return image_rgba[60:144,86:170,0:3].astype(float)
        
        return np.zeros((84,84,3)).astype(float)
        """
    # Computes the reward functinon based on collision.
    def __compute_reward(self, collision_info, car_state, cov_reward, action):

        MAX_SPEED = 2.5
        alpha = 1.0

        # If the car has collided, the reward is always zero
        if (collision_info.has_collided):
            return 0.0

        # If the car has stopped for some reason, the reward is always zero
        if abs(car_state.speed) < 0.02:
            return 0.0

        # If there is no new coverage, there is no reward
        if cov_reward < 0.1:
                return 0.0 

        # straight will be rewarded as 1.0, semi straight as 0.5
        direction_reward = float(2 - abs(action - 2)) / 2.0

        # final reward
        reward = alpha * cov_reward + (1 - alpha) * direction_reward

        #print("cov reward: {}, reward: {}".format(cov_reward, reward))
        return reward

    # prepare starting points list
    def __get_starting_points(self):
        
        starting_points_file = open(os.path.join(self.__data_dir,'data\\starting_points.txt'))
        starting_points_list = []
        
        for line in starting_points_file:
            starting_points_list.append([float(x) for x in line.split(' ')[:3]])
        
        return starting_points_list

    # get most newly generated random point
    def __get_next_generated_random_point(self):
        
        # grab the newest line with generated random point
        newest_rp = "None"

        # keep searching until the simulation is giving something
        while newest_rp == "None":
            
            # notify user
            print("Searching for a random point...")

            # open log file
            log_file = open(self.__log_file, "r")

            # search for the newest generated random point line
            for line in log_file:
                if "RandomPoint" in line:
                    newest_rp = line
            
        # notify user
        print("Found random point.")
        
        # filter random point from line
        random_point = [float(newest_rp.split(" ")[-3].split("=")[1]), float(newest_rp.split(" ")[-2].split("=")[1]), float(newest_rp.split(" ")[-1].split("=")[1])]
        return random_point

        #idx = randint(0, len(self.__starting_points)-1)
        #return self.__starting_points[idx]

    # Randomly selects a starting point on the road
    # Used for initializing an iteration of data generation from AirSim
    def __get_next_starting_point(self):
    
        # get random start point from log file, and make it relative to agent's starting point
        random_start_point = self.__get_next_generated_random_point()
        random_start_point = [random_start_point[0]-self.__start_point[0], random_start_point[1]-self.__start_point[1], random_start_point[2]-self.__start_point[2]]
        random_start_point = [x / 100.0 for x in random_start_point]

        # draw random orientation
        random_direction = (0, 0, np.random.uniform(-math.pi,math.pi))

        # Get the current state of the vehicle
        car_state = self.__car_client.getCarState()

        # The z coordinate is always zero
        random_start_point[2] = -0
        return (random_start_point, random_direction)

    # A helper function to make a directory if it does not exist
    def __make_dir_if_not_exist(self, directory):
        if not (os.path.exists(directory)):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

# convert euler angle to quaternion
def toQuaternion(pitch, roll, yaw):
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    q = airsim.Quaternionr()
    q.w_val = t0 * t2 * t4 + t1 * t3 * t5 #w
    q.x_val = t0 * t3 * t4 - t1 * t2 * t5 #x
    q.y_val = t0 * t2 * t5 + t1 * t3 * t4 #y
    q.z_val = t1 * t2 * t4 - t0 * t3 * t5 #z
    return q

def transform_depth_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)

    if img1d.size > 1:

        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = PIL.Image.fromarray(img2d)

        # debug only
        #image_png = image.convert('RGB')
        #image_png.save("DistributedRL\\debug\\{}.png".format(time.time()))

        im_final = np.array(image.resize((84, 84)).convert('L')) 
        im_final = im_final / 255.0

        return im_final

    return np.zeros((84,84)).astype(float)


# Sets up the logging framework.
# This allows us to log using simple print() statements.
# The output is redirected to a unique file on the file share.
def setup_logs(parameters):
    output_dir = 'Z:\\logs\\{0}\\agent'.format(parameters['experiment_name'])
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    sys.stdout = open(os.path.join(output_dir, '{0}.stdout.txt'.format(os.environ['AZ_BATCH_NODE_ID'])), 'w')
    sys.stderr = open(os.path.join(output_dir, '{0}.stderr.txt'.format(os.environ['AZ_BATCH_NODE_ID'])), 'w')

# Parse the command line parameters
parameters = {}
for arg in sys.argv:
    if '=' in arg:
        args = arg.split('=')
        print('0: {0}, 1: {1}'.format(args[0], args[1]))
        parameters[args[0].replace('--', '')] = args[1]
    if arg.replace('-', '') == 'local_run':
        parameters['local_run'] = True

#Make the debug statements easier to read
#np.set_printoptions(threshold=np.nan, suppress=True)

# Manually add parameters
"""
if 'batch_update_frequency' not in parameters.keys(): 
    parameters['batch_update_frequency'] = 300
if 'max_epoch_runtime_sec' not in parameters.keys(): 
    parameters['max_epoch_runtime_sec'] = 30
if 'per_iter_epsilon_reduction' not in parameters.keys(): 
    parameters['per_iter_epsilon_reduction'] = 0.003
if 'min_epsilon' not in parameters.keys(): 
    parameters['min_epsilon'] = 0.1
if 'batch_size' not in parameters.keys(): 
    parameters['batch_size'] = 32
if 'replay_memory_size' not in parameters.keys(): 
    parameters['replay_memory_size'] = 300
if 'train_conv_layers' not in parameters.keys(): 
    parameters['train_conv_layers'] = 'false'
if 'airsim_path' not in parameters.keys(): 
    parameters['airsim_path'] = 'E:\\AD_Cookbook_AirSim\\'
if 'data_dir' not in parameters.keys(): 
    parameters['data_dir'] = os.path.join(os.getcwd(), 'DistributedRL\\Share')
if 'experiment_name' not in parameters.keys(): 
    parameters['experiment_name'] = 'local_run'
if 'log_path' not in parameters.keys(): 
    parameters['log_path'] = "..\\..\\Unreal Projects\\Building99\\Saved\\Logs\\Building_99.log"
if 'local_run' not in parameters.keys(): 
    parameters['local_run'] = 'true'
"""
if 'exp_type' not in parameters.keys():
    parameters['exp_type'] = 'with_rgb'
if 'start_x' not in parameters.keys(): 
    parameters['start_x'] = 500.0
if 'start_y' not in parameters.keys(): 
    parameters['start_y'] = 850.0
if 'start_z' not in parameters.keys(): 
    parameters['start_z'] = 32.0
if 'log_path' not in parameters.keys(): 
    parameters['log_path'] = "D:\\AD_Cookbook_AirSim\\Building99\\Building_99\\Saved\\Logs\\Building_99.log"


# Check additional parameters needed for local run
if 'local_run' in parameters:
    if 'airsim_path' not in parameters:
        print('ERROR: for a local run, airsim_path must be defined.')
        print('Please provide the path to airsim in a parameter like "airsim_path=<path_to_airsim>"')
        print('It should point to the folder containing AD_Cookbook_Start_AirSim.ps1')
        sys.exit()
    if 'batch_update_frequency' not in parameters:
        print('ERROR: for a local run, batch_update_frequency must be defined.')
        print('Please provide the path to airsim in a parameter like "batch_update_frequency=<int>"')
        sys.exit()

# Set up the logging to the file share if not running locally.
if 'local_run' not in parameters:
    setup_logs(parameters)

print('------------STARTING AGENT----------------')
print(parameters)

print('***')
print(os.environ)
print('***')

# Identify the node as an agent and start AirSim
if 'local_run' not in parameters:
    os.system('echo 1 >> D:\\agent.agent')
    os.system('START "" powershell.exe D:\\AD_Cookbook_AirSim\\Scripts\\DistributedRL\\restart_airsim_if_agent.ps1')
else:
    os.system('START "" powershell.exe {0}'.format(os.path.join(parameters['airsim_path'], 'AD_Cookbook_Start_AirSim.ps1 neighborhood -windowed')))
    
# Start the training
agent = DistributedAgent(parameters)
agent.start()
