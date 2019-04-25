import airsim
import cv2
import numpy as np
import math
from scipy.ndimage import rotate
import time
from PIL import Image
import pptk
import copy

class CoverageMap:

    # initiate coverage map
    def __init__(self, start_point, map_size, state_size, input_size, height_threshold):

        self.start_point = start_point # agent's starting point from the simulation, in centimeters
        self.map_size = map_size # map size to be in the shape of [map_size, map_size], in centimeters
        self.state_size = state_size # state size to be in the shape of [state_size, state_size], in centimeters
        self.input_size = input_size # final input size to be in the shape of [input_size, input_size], in centimeters
        self.height_threshold = height_threshold # height_threshold, in meters
        self.saved_state = None # save state in each iteration

        # prepare clean coverage map
        self.cov_map = np.zeros((self.map_size, self.map_size))
        self.previous_cov_map = copy.deepcopy(self.cov_map)

    # clear coverage map and state
    def reset(self):

        self.saved_state = None
        self.cov_map = np.zeros((self.map_size, self.map_size))

    # set airsim client, to get position, orientation and lidar data
    def set_client(self, client):

        self.client = client 

    # get the amount of new terain reveiled by the agent
    def get_progress(self):

        return np.subtract(self.cov_map, self.previous_cov_map).sum()

    # get the entire map, rescaled to the input size
    def get_map_scaled(self):

        # TODO: find a better method to resize
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

        # update previous coverage map
        self.previous_cov_map = copy.deepcopy(self.cov_map)

        # get car position. convert it to be relative to the world in centimeters
        client_pose = self.client.simGetVehiclePose()
        pose = [int(self.start_point[0] + (self.map_size / 2) - (client_pose.position.x_val * 100.0)), 
                int(self.start_point[1] + (self.map_size / 2) + (client_pose.position.y_val * 100.0)),
                int(self.start_point[2] + (self.map_size / 2) + (client_pose.position.z_val * 100.0))]
        
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
            points_trimmed[:,0] = np.subtract(pose[0],points_trimmed[:,0] * 100.0)
            points_trimmed[:,1] = np.add(pose[1],points_trimmed[:,1] * 100.0)
            points_trimmed = points_trimmed.astype(int)
            
            # paint selected indexes
            for i in range(points_trimmed.shape[0]):
                self.cov_map[points_trimmed[i][0],points_trimmed[i][1]] = 200

            # extract state from nav map
            x_range = (int(pose[0] - self.state_size/2), int(pose[0] + self.state_size/2))
            y_range = (int(pose[1] - self.state_size/2), int(pose[1] + self.state_size/2))
            state = self.cov_map[x_range[0]:x_range[1],y_range[0]:y_range[1]]
            
            # make it more visible
            #idxs = np.where(state > 0.0)
            #state[idxs] = 255.0

            # scale using PIL
            im = Image.fromarray(np.uint8(state))
            im = im.resize((self.input_size*2, self.input_size*2), Image.ANTIALIAS)

            # rotate according to the orientation
            im = im.rotate(math.degrees(angles[2]))
            state = np.array(im)

            # extract half of the portion to receive state in input size, save it for backup
            self.saved_state = state[int(state.shape[0]/2 - state.shape[0]/4):int(state.shape[0]/2 + state.shape[0]/4), 
                            int(state.shape[1]/2 - state.shape[1]/4):int(state.shape[1]/2 + state.shape[1]/4)]
            
            return self.saved_state



if __name__ == "__main__":

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()

    start_point = [500.0, 850.0, 32.0]

    covMap = CoverageMap(start_point=start_point, map_size=10000, state_size=2000, input_size=84, height_threshold=0.9)
    covMap.set_client(client=client)

    while True:
        startTime = time.time()

        state = covMap.get_state()
        #maps = covMap.get_map_scaled()

        reward = covMap.get_progress()

        cv2.imshow('navigation map (q to exit)', state)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        endTime = time.time()
        # present fps
        print("reward: {}, fps: {}".format(reward, 1/(endTime-startTime)))
        

    cv2.destroyAllWindows()

