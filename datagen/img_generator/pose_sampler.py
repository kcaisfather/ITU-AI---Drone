import cv2
import numpy as np
import os
import sys
from os.path import isfile, join
import airsimdroneracingvae as airsim
#import airsim

# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import DrivetrainType, Pose, Vector3r, Quaternionr
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
from scipy.integrate._ivp.radau import P
from scipy.spatial.transform import Rotation
import time

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)

# Extras for Perception
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image

import Dronet
import lstmf

#Extras for Trajectory and Control
import pickle
import random
from numpy import zeros
from joblib import dump, load

from quadrotor import *
from network import Net, Net_Regressor


class PoseSampler:
    def __init__(self, dataset_path, flight_log, parcour, with_gate=True):
        self.num_samples = 1
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'files/gate_training_data.csv')
        self.curr_idx = 0
        self.current_gate = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(1)        
        self.configureEnvironment()

        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)

        self.log_path = os.path.join(self.base_path, 'files/flight_log.txt')
        self.flight_log = flight_log
        self.parcour = parcour

        #----- Classifier/Regressor parameters -----------------------------
        self.mp_classifier = Net()
        self.t_or_s_classifier = Net()
        self.speed_regressor = Net_Regressor()
 

        self.test_covariances = {"MAX_SAFE":[], "MAX_NO_SAFE":[], "DICE_SAFE" :[], "DICE_NO_SAFE" :[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_jerk_full_stop":[]}
        

        #---- Model import ---------------------------------
        self.device = torch.device("cpu")

        input_size = 4
        output_size = 4
        lstmR_hidden_size = 16
        lstmR_num_layers = 1

        # Dronet
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load(self.base_path + '/weights/Dronet_new.pth',map_location=torch.device('cpu')))   
        self.Dronet.eval()

        # LstmR
        self.lstmR = lstmf.LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(self.device)
        #print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load(self.base_path + '/weights/R_2.pth',map_location=torch.device('cpu')))   
        self.lstmR.eval() 


        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.period_denum = 30.

        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            #transforms.Lambda(self.gaussian_blur),
                            #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                            transforms.ToTensor()])
        
        rand_x = []
        rand_y = []
        rand_z = []

        for i in range (16):
            rand_x.append(uniform(-0.15,0.15))
            rand_y.append(uniform(-0.15,0.15))
            rand_z.append(uniform(-0.15,0.15))
        

        if parcour == "hexa": # hexagon
            quat0 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
            quat1 = R.from_euler('ZYX',[30.,0.,0.],degrees=True).as_quat()
            quat2 = R.from_euler('ZYX',[-30.,0.,0.],degrees=True).as_quat()
            quat3 = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
            quat4 = R.from_euler('ZYX',[-150.,0.,0.],degrees=True).as_quat() 
            quat5 = R.from_euler('ZYX',[-210.,0.,0.],degrees=True).as_quat() 
            self.track = [Pose(Vector3r(10.,25.,-2.) + Vector3r(rand_x[0],rand_y[0],rand_z[0]), Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                        Pose(Vector3r(25.,15.,-1) + Vector3r(rand_x[1],rand_y[1],rand_z[1]), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                        Pose(Vector3r(25.,5.,-1.5) + Vector3r(rand_x[2],rand_y[2],rand_z[2]), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                        Pose(Vector3r(10.,-5,-3) + Vector3r(rand_x[3],rand_y[3],rand_z[3]), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                        Pose(Vector3r(-5.,5.,-3) + Vector3r(rand_x[4],rand_y[4],rand_z[4]), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                        Pose(Vector3r(-5.,15.,-2) + Vector3r(rand_x[5],rand_y[5],rand_z[5]), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3]))]


            quat_drone = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            self.drone_init = Pose(Vector3r(5.,25.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))
        
        
        elif parcour == "eight": # Eight - Infinity
            quat0 = R.from_euler('ZYX',[55.,0.,0.],degrees=True).as_quat()
            quat1 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat()
            quat2 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            quat3 = R.from_euler('ZYX',[-45.,0.,0.],degrees=True).as_quat()
            quat4 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()  
            quat5 = R.from_euler('ZYX',[-45.,0.,0.],degrees=True).as_quat() 
            quat6 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat() 
            quat7 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat() 
            quat8 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat() 
            quat9 = R.from_euler('ZYX',[-45.,0.,0.],degrees=True).as_quat() 
            quat10 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()  
            quat11 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat()
            quat12 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()  
            quat13 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat() 
            quat14 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat() 
            quat15 = R.from_euler('ZYX',[-45.,0.,0.],degrees=True).as_quat() 
            
            self.track = [Pose(Vector3r(1. ,18.,-2.) + Vector3r(rand_x[0],rand_y[0],rand_z[0]), Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                        Pose(Vector3r(5. ,15.,-1.) + Vector3r(rand_x[1],rand_y[1],rand_z[1]), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                        Pose(Vector3r(9. ,10. ,-1.5) + Vector3r(rand_x[2],rand_y[2],rand_z[2]), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                        Pose(Vector3r(5. ,5. ,-2) + Vector3r(rand_x[3],rand_y[3],rand_z[3]), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                        Pose(Vector3r(0.,0.,-1) + Vector3r(rand_x[4],rand_y[4],rand_z[4]), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                        Pose(Vector3r(-5.,-5.,-1.5) + Vector3r(rand_x[5],rand_y[5],rand_z[5]), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3])),
                        Pose(Vector3r(-9.,-10.,-2.5) + Vector3r(rand_x[6],rand_y[6],rand_z[6]), Quaternionr(quat6[0],quat6[1],quat6[2],quat6[3])),
                        Pose(Vector3r(-5.,-15.,-3) + Vector3r(rand_x[7],rand_y[7],rand_z[7]), Quaternionr(quat7[0],quat7[1],quat7[2],quat7[3])),
                        Pose(Vector3r(0.,-18.,-2) + Vector3r(rand_x[8],rand_y[8],rand_z[8]), Quaternionr(quat8[0],quat8[1],quat8[2],quat8[3])),
                        Pose(Vector3r(5.,-15.,-1) + Vector3r(rand_x[9],rand_y[9],rand_z[9]), Quaternionr(quat9[0],quat9[1],quat9[2],quat9[3])),
                        Pose(Vector3r(9.,-10.,-1.5) + Vector3r(rand_x[10],rand_y[10],rand_z[10]), Quaternionr(quat10[0],quat10[1],quat10[2],quat10[3])),
                        Pose(Vector3r(5.,-5.,-2) + Vector3r(rand_x[11],rand_y[11],rand_z[11]), Quaternionr(quat11[0],quat11[1],quat11[2],quat11[3])), 
                        Pose(Vector3r(0.,0.,-3) + Vector3r(rand_x[12],rand_y[12],rand_z[12]), Quaternionr(quat12[0],quat12[1],quat12[2],quat12[3])),
                        Pose(Vector3r(-5.,5.,-2) + Vector3r(rand_x[13],rand_y[13],rand_z[13]), Quaternionr(quat13[0],quat13[1],quat13[2],quat13[3])),
                        Pose(Vector3r(-9.,10.,-3) + Vector3r(rand_x[14],rand_y[14],rand_z[14]), Quaternionr(quat14[0],quat14[1],quat14[2],quat14[3])),
                        Pose(Vector3r(-5.,15.,-2.5) + Vector3r(rand_x[15],rand_y[15],rand_z[15]), Quaternionr(quat15[0],quat15[1],quat15[2],quat15[3]))]
            
            quat_drone = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            self.drone_init = Pose(Vector3r(0.,30.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))


        elif parcour == "circle": # Circle
            quat0 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
            quat1 = R.from_euler('ZYX',[60.,0.,0.],degrees=True).as_quat()
            quat2 = R.from_euler('ZYX',[30.,0.,0.],degrees=True).as_quat()
            quat3 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            quat4 = R.from_euler('ZYX',[-30.,0.,0.],degrees=True).as_quat() 
            quat5 = R.from_euler('ZYX',[-60.,0.,0.],degrees=True).as_quat() 
            quat6 = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
            quat7 = R.from_euler('ZYX',[-120.,0.,0.],degrees=True).as_quat()
            quat8 = R.from_euler('ZYX',[-150.,0.,0.],degrees=True).as_quat()
            quat9 = R.from_euler('ZYX',[-180.,0.,0.],degrees=True).as_quat()
            quat10 = R.from_euler('ZYX',[-210.,0.,0.],degrees=True).as_quat()
            quat11 = R.from_euler('ZYX',[-250.,0.,0.],degrees=True).as_quat()
            self.track = [Pose(Vector3r(0.,10.,-2.) , Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                        Pose(Vector3r(5.,8.66,-2) , Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                        Pose(Vector3r(8.66,5.,-2) , Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                        Pose(Vector3r(10.,0.,-2) , Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                        Pose(Vector3r(8.66,-5.,-2) , Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                        Pose(Vector3r(5.,-8.66,-2) , Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3])), 
                        Pose(Vector3r(0.,-10.,-2) , Quaternionr(quat6[0],quat6[1],quat6[2],quat6[3])),
                        Pose(Vector3r(-5.,-8.66,-2) , Quaternionr(quat7[0],quat7[1],quat7[2],quat7[3])),
                        Pose(Vector3r(-8.66,-5,-2) , Quaternionr(quat8[0],quat8[1],quat8[2],quat8[3])),
                        Pose(Vector3r(-10.,0,-2) , Quaternionr(quat9[0],quat9[1],quat9[2],quat9[3])),
                        Pose(Vector3r(-8.66,5.,-2) , Quaternionr(quat10[0],quat10[1],quat10[2],quat10[3])),
                        Pose(Vector3r(-5.,8.66,-2) , Quaternionr(quat11[0],quat11[1],quat11[2],quat11[3]))]
            
            quat_drone = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            self.drone_init = Pose(Vector3r(-5.,10.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))

        elif self.parcour == "spline": # Spline
            quat0 = R.from_euler('ZYX',[0.,0.,0.] ,degrees=True).as_quat()
            quat1 = R.from_euler('ZYX',[20.,0.,0.],degrees=True).as_quat()
            quat2 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
            quat3 = R.from_euler('ZYX',[-15.,0.,0.],degrees=True).as_quat()
            quat4 = R.from_euler('ZYX',[-45.,0.,0.],degrees=True).as_quat()
            quat5 = R.from_euler('ZYX',[20.,0.,0.],degrees=True).as_quat()
            self.track = [Pose(Vector3r(0.,20.,-2.) + Vector3r(rand_x[0],rand_y[0],rand_z[0]) , Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                        Pose(Vector3r(2.,10.,-1) + Vector3r(rand_x[1],rand_y[1],rand_z[1]), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                        Pose(Vector3r(3.,0.,-1.5) + Vector3r(rand_x[2],rand_y[2],rand_z[2]), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                        Pose(Vector3r(2.,-5.,-3) + Vector3r(rand_x[3],rand_y[3],rand_z[3]), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                        Pose(Vector3r(0.,-10.,-3) + Vector3r(rand_x[4],rand_y[4],rand_z[4]), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                        Pose(Vector3r(-1.,-25.,-2) + Vector3r(rand_x[5],rand_y[5],rand_z[5]), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3]))]
            
            quat_drone = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
            self.drone_init = Pose(Vector3r(0.,30.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))

            
            
            #Pose(Vector3r(30.,5.,-1) + Vector3r(rand_x[2],rand_y[2],rand_z[2]), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
            #Pose(Vector3r(10.,-5,-2) + Vector3r(rand_x[3],rand_y[3],rand_z[3]), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
            #Pose(Vector3r(-5.,5.,-2) + Vector3r(rand_x[4],rand_y[4],rand_z[4]), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
            #Pose(Vector3r(-5.,15.,-2.5) + Vector3r(rand_x[5],rand_y[5],rand_z[5]), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3]))
            

        #-----------------------------------------------------------------------             
    


    def test_algorithm(self, method = "MAX", use_model = False, safe_mode = True, time_or_speed = 1, v_average = 1.):
        
        pose_prediction = np.zeros((1000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)


        while not self.client.isApiControlEnabled():
            self.client.enableApiControl(True)
        
        ApiControlCheck = self.client.isApiControlEnabled()

        #if drone is at initial point
        self.client.simSetVehiclePose(self.drone_init, True)
        self.client.armDisarm(True)
        
        time.sleep(2)

        print("Api control enabled: ",ApiControlCheck)
        
        if self.parcour == "circle":
            self.client.moveToPositionAsync(-3,10,-2.2,1,drivetrain=DrivetrainType.ForwardOnly).join()
            path_step = 2
        elif self.parcour == "spline":
            self.client.moveToPositionAsync(0,25,-2.3,1,drivetrain=DrivetrainType.ForwardOnly).join()
            path_step = 4
        elif self.parcour == "hexa":
            self.client.moveToPositionAsync(7,25,-2.3,1,drivetrain=DrivetrainType.ForwardOnly).join()
            path_step = 3
        
        self.curr_idx = 0
        path_index = 0
    
        track_completed = False
        fail_check = False


        covariance_sum = 0.
        prediction_std = [0., 0., 0., 0.]
        sign_coeff = 0. 
        covariance_list = []
        cov_rep_num = 5


        while((not track_completed) and (not fail_check)):            

            sign_coeff = 1.

            self.brightness = 0.
            self.contrast = 0.
            self.saturation = 0.
            self.transformation = transforms.Compose([
                    transforms.Resize([200, 200]),
                    #transforms.Lambda(self.gaussian_blur),
                    #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                    transforms.ToTensor()])
                
            noise_coeff = self.brightness + self.contrast + self.saturation


            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            # anyGate = self.isThereAnyGate(img_rgb)
            #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
            img =  Image.fromarray(img_rgb)
            image = self.transformation(img)
            
            with torch.no_grad():   
                # Determine Gate location with Neural Networks
                pose_gate_body = self.Dronet(image)
                predicted_r = np.copy(pose_gate_body[0][0])


                if predicted_r < 3.0:
                    self.period_denum = 3.0
                elif predicted_r < 5.0:
                    self.period_denum = 3.0
                else:
                    self.period_denum = 5.0


                for i,num in enumerate(pose_gate_body.reshape(-1,1)):
                    #print(num, i , self.curr_idx)
                    pose_prediction[self.curr_idx][i] = num.item()

                if self.curr_idx >= 11:
                    pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
                    for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                        prediction_std[i] = p_g_c.item()

                    prediction_std = np.clip(prediction_std, 0, prediction_std)
                    prediction_std = prediction_std.ravel()
                    covariance_sum = np.sum(prediction_std)

                    self.test_covariances[method].append(covariance_sum)

                    covariance_list.append(covariance_sum)
                    if self.curr_idx >= (11 + cov_rep_num):
                        covariance_sum = np.sum(covariance_list[-cov_rep_num:]) / float(cov_rep_num)

                    # Gate ground truth values will be implemented
                    pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
                
                    # Trajectory generate

                    quadpose = self.client.simGetVehiclePose()

                    #rpy = R.from_euler("zyx",[quadpose.orientation.w_val,quadpose.orientation.x_val,quadpose.orientation.y_val,quadpose.orientation.z_val]).as_euler('zyx')

                    roll, pitch, yaw = self.euler_from_quaternion(quadpose.orientation.x_val,quadpose.orientation.y_val,quadpose.orientation.z_val,quadpose.orientation.w_val)
                    drone_pos = [quadpose.position.x_val,quadpose.position.y_val,quadpose.position.z_val,roll,pitch,yaw]

                    waypoint_world = spherical_to_cartesian(drone_pos, pose_gate_body)

                    print("Waypoint Worlds: ",waypoint_world)

                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]                    
                    yaw_diff = pose_gate_body[3][0]
                    self.yawf = (drone_pos[5]+yaw_diff) + np.pi/2

                    print("\nCurrent index: {0}".format(self.curr_idx))
                    print("Predicted r: {0:.3}, Noise coeff: {1:.4}, Covariance sum: {2:.3}".format(pose_gate_body[0][0], sign_coeff*noise_coeff, covariance_sum))
                    print("Predicted Gate Location: ", posf)
                    #print("True Gate Location: ", self.gate[0].position)
                    print("True Quad Pose: ", drone_pos)
                    print("yawf: ", self.yawf)

                    x_gate = np.float32(posf[0]).item()
                    y_gate = np.float32(posf[1]).item()
                    z_gate = np.float32(posf[2]).item()

                    x_target = drone_pos[0] + ((x_gate-drone_pos[0])/2)
                    y_target = drone_pos[1] + ((y_gate-drone_pos[1])/2)
                    z_target = drone_pos[2] + ((z_gate-drone_pos[2])/2)

                    if path_index % path_step == 1:
                        x_target = posf[0]
                        y_target = posf[1]
                        z_target = posf[2]

                    print("The point drone is going: ", x_target, " , ", y_target," , ",z_target)
                    self.client.moveToPositionAsync(x_target,y_target,z_target,1.2,drivetrain=DrivetrainType.ForwardOnly).join()

                    path_index += 1
                        
                    
            self.curr_idx += 1


    def euler_from_quaternion(self,x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


    def update(self, mode):
        '''
        convetion of names:
        p_a_b: pose of frame b relative to frame a
        t_a_b: translation vector from a to b
        q_a_b: rotation quaternion from a to b
        o: origin
        b: UAV body frame
        g: gate frame
        '''

        # create and set pose for the quad
        #p_o_b, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
        
        # create and set gate pose relative to the quad
        #p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        #self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        #min_vel, min_acc, min_jerk, pos_waypoint_interp, min_acc_stop, min_jerk_full_stop
        MP_list = ["min_acc", "min_jerk", "min_jerk_full_stop", "min_vel"]
        #MP_list = ["min_vel"]

        if self.with_gate:
            # gate_name = "gate_0"
            # self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            # self.client.simSetObjectPose(self.tgt_name, self.track[0], True)
            for i, gate in enumerate(self.track):
                #print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), Vector3r(0.75,0.75,0.75))
                #self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)
        # request quad img from AirSim
        time.sleep(0.001)




        if mode == "TEST":    
            self.mp_classifier.load_state_dict(torch.load(self.base_path + '/classifier_files/best_2.pt'))
            #self.time_regressor = load(self.base_path + '/classifier_files/dt_regressor.sav')
            self.time_coeff = 1.5
            v_average = 1.5 
            #self.mp_scaler = load(self.base_path + 'classifier_files/mp_scaler.bin')
            #self.time_scaler = load(self.base_path + 'classifier_files/time_scaler.bin')
            print("\n>>> PREDICTION MODE: DICE, SAFE MODE: ON")
            self.test_algorithm(use_model=True, method="DICE_SAFE", safe_mode = True, time_or_speed = 0, v_average = v_average)
            
            for method in MP_list:
                print("\n>>> TEST MODE: " + method)
                self.test_algorithm(method = method, time_or_speed = 0, v_average = v_average)

            self.test_number = "0_0"
            pickle.dump([self.test_states,self.test_arrival_time,self.test_costs, self.test_safe_counter, self.test_distribution_on_noise, self.test_distribution_off_noise, self.test_covariances, self.test_methods], open(self.base_path + "files/test_variables_" + self.test_number + ".pkl","wb"), protocol=2)
        

        elif mode == "VISUAL":
            self.visualize_drone()
        else:
            print("There is no such a mode called " + "'" + mode + "'")

    
    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), Vector3r(0.75,0.75,0.75))
            #self.tgt_name = self.client.simSpawnObject("gate", "Gate", Pose(position_val=Vector3r(0,0,15)), Vector3r(4,4,4), physics_enabled=False)
        else:
            self.tgt_name = "empty_target"

        if os.path.exists(self.csv_path):
            self.file = open(self.csv_path, "a")
        else:
            self.file = open(self.csv_path, "w")

    