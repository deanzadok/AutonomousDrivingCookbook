import setup_path
import airsim
import cv2
import numpy as np
import math
import time
from PIL import Image
import copy

class CoverageMap:

    # initiate coverage map
    def __init__(self, start_point, map_size, scale_ratio, state_size, input_size, height_threshold, reward_norm):

        self.start_point = start_point # agent's starting point from the simulation, in centimeters
        self.map_size = map_size # map size to be in the shape of [map_size, map_size], in centimeters
        self.scale_ratio = scale_ratio # scale ratio to be used to reduce the map size and increase performance
        self.state_size = int(state_size / self.scale_ratio) # state size to be in the shape of [state_size, state_size], in centimeters
        self.input_size = input_size # final input size to be in the shape of [input_size, input_size], in centimeters
        self.height_threshold = height_threshold # height_threshold, in meters
        self.saved_state = None # save state in each iteration
        self.reward_norm = reward_norm # factor to normalize the reward

        # prepare clean coverage map
        self.cov_map = np.zeros((int(self.map_size / self.scale_ratio), int(self.map_size / self.scale_ratio)))

    # clear coverage map and state
    def reset(self):

        self.saved_state = None
        self.cov_map = np.zeros((int(self.map_size / self.scale_ratio), int(self.map_size / self.scale_ratio)))

    # set airsim client, to get position, orientation and lidar data
    def set_client(self, client):

        self.client = client

    # get the entire map, rescaled to the input size
    def get_map_scaled(self):

        # TODO: resize without PIL
        # resize coverage image using PIL
        im = Image.fromarray(np.uint8(self.cov_map))
        im = im.resize((self.input_size,self.input_size), Image.BILINEAR)

        # make it binary
        binary_map = np.array(im)
        idxs = np.where(binary_map > 0.0)
        binary_map[idxs] = 255.0

        return binary_map

    # get state image from the coverage map
    def get_state(self):

        # get car position, convert it to be relative to the world in centimeters, and convert it according to scale ratio
        client_pose = self.client.simGetVehiclePose()
        pose = [int(round((self.start_point[0] + (self.map_size / 2) - (client_pose.position.x_val * 100.0))/self.scale_ratio)), 
                int(round((self.start_point[1] + (self.map_size / 2) + (client_pose.position.y_val * 100.0))/self.scale_ratio)),
                int(round((self.start_point[2] + (self.map_size / 2) + (client_pose.position.z_val * 100.0))/self.scale_ratio))]

        # get car orientation
        angles = airsim.to_eularian_angles(client_pose.orientation)

        # get lidar data
        lidarData = self.client.getLidarData(lidar_name='LidarSensor1', vehicle_name='Car')
        if (len(lidarData.point_cloud) < 3):
            print("\tNo points received from Lidar data")
            return self.saved_state
        else:
            # reshape array of floats to array of [X,Y,Z]
            points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))

            # trim high points
            idxs = np.where(points[:,2] < self.height_threshold)[0]
            points_trimmed = np.delete(points, idxs, axis=0)

            # rotate it according to the current orientation
            rot_angle = angles[2] + math.pi
            rot_matrix = np.array([[-math.cos(rot_angle), -math.sin(rot_angle), 0],
                                    [math.sin(rot_angle), -math.cos(rot_angle), 0],
                                    [0, 0, 1]])
            points_trimmed = np.dot(points_trimmed, rot_matrix)

            # convert it to be relative to the world in centimeters, z axis is not relevant 
            points_trimmed[:,0] = np.subtract(pose[0],np.rint(points_trimmed[:,0] * 100.0 / self.scale_ratio))
            points_trimmed[:,1] = np.add(pose[1],np.rint(points_trimmed[:,1] * 100.0 / self.scale_ratio))
            points_trimmed = points_trimmed.astype(int)
            
            # paint selected indexes, and sum new pixels
            new_pixels = 0
            for i in range(points_trimmed.shape[0]):
                if self.cov_map[points_trimmed[i][0],points_trimmed[i][1]] == 0:
                    self.cov_map[points_trimmed[i][0],points_trimmed[i][1]] = 255
                    new_pixels += 1

            # extract state from nav map
            x_range = (int(pose[0] - self.state_size/2), int(pose[0] + self.state_size/2))
            y_range = (int(pose[1] - self.state_size/2), int(pose[1] + self.state_size/2))
            state = self.cov_map[x_range[0]:x_range[1],y_range[0]:y_range[1]]

            # scale using PIL
            im = Image.fromarray(np.uint8(state))
            im = im.resize((self.input_size*2, self.input_size*2), Image.ANTIALIAS)

            # rotate according to the orientation
            im = im.rotate(math.degrees(angles[2]))
            state = np.array(im)

            # extract half of the portion to receive state in input size, save it for backup
            self.saved_state = state[int(state.shape[0]/2 - state.shape[0]/4):int(state.shape[0]/2 + state.shape[0]/4), 
                            int(state.shape[1]/2 - state.shape[1]/4):int(state.shape[1]/2 + state.shape[1]/4)]
            
            # send state as input and reward
            return self.saved_state, min(new_pixels / self.reward_norm, 1.0)



if __name__ == "__main__":

    # connect to AirSim 
    client = airsim.CarClient()
    client.confirmConnection()

    # create coverage map and connect to client
    start_point = [500.0, 850.0, 32.0]
    covMap = CoverageMap(start_point=start_point, map_size=12000, scale_ratio=1, state_size=2000, input_size=84, height_threshold=0.95, reward_norm=1000.0)
    covMap.set_client(client=client)

    # start free run session
    i = 1
    fps_sum = 0
    while True:
        startTime = time.time()

        # get state and show it on screen
        state, reward = covMap.get_state()
        #state = covMap.get_map_scaled()

        cv2.imshow('navigation map (q to exit)', state)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        endTime = time.time()

        # present fps or reward
        fps_sum += (1/(endTime-startTime))
        #print("fps average: %.2f" % (fps_sum/i))
        if i % 3 == 0:
            print("reward: {}".format(reward))

        i+=1
        
    cv2.destroyAllWindows()
