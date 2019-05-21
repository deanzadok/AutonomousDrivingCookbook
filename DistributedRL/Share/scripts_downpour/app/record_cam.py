import setup_path 
import airsim
import os
import numpy as np
import pdb
import time
import datetime
import pprint
import keyboard  # using module keyboard
import math
from PIL import Image
from coverage_map import CoverageMap
	
# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
car_controls = airsim.CarControls()

# create coverage map and connect to client
start_point = [-1200.0, -500.0, 62.000687]
covMap = CoverageMap(start_point=start_point, map_size=12000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
covMap.set_client(client=client)

# create experiments directories
experiment_dir = os.path.join(os.path.expanduser('~'), 'Documents\AirSim', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
images_dir = os.path.join(experiment_dir, 'images')
os.makedirs(images_dir)

# create txt file
airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w") 
airsim_rec.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tImageFile\n") 

idx = 0
while True:

    time_stamp = int(time.time()*10000000)

    # get image from AirSim
    image_response = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    if image1d.size > 1:
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    else:
        image_rgba = np.zeros((144,259,3)).astype(float)
    
    # get coverage image and put it inside the RGB image
    cov_image, _ = covMap.get_state_from_pose()
    cov_image = np.expand_dims(cov_image, axis=2)
    cov_image = np.repeat(cov_image, 4, axis=2)
    image_rgba[:cov_image.shape[0],:cov_image.shape[1],:] = cov_image

    # save the combined image
    im = Image.fromarray(np.uint8(image_rgba))
    im.save(os.path.join(images_dir, "ptc_{}.png".format(time_stamp)))
	
    # get position and car state
    client_pose = client.simGetVehiclePose()
    car_state = client.getCarState()

    # write meta-date to text file
    airsim_rec.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time_stamp,client_pose.position.x_val,client_pose.position.y_val,client_pose.position.z_val,car_state.rpm,car_state.speed,"ptc_{}.png".format(time_stamp))) 
		
    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        airsim_rec.close()
        quit()
	
    idx += 1
    time.sleep(0.2)

airsim_rec.close() 