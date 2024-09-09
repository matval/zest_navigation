#!/usr/bin/env python3
import cv2
import copy
import torch
import numpy as np
import torchvision.transforms as transforms

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.travnet import TravNet

class Object(object):
    pass

class TRAVERSABILITY_NODE:
    def __init__(self):
        rospy.init_node('traversability_map_predictor')
        self.model_path = rospy.get_param('~model_path', '/home/gian/Documents/Códigos/TCC/octo_planning/src/octomap_utils/checkpoints/best_wayfast.pth')
        self.image_topic = rospy.get_param('~image_topic', '/terrasentia/zed2/zed_node/rgb/image_rect_color/compressed')
        self.depth_topic = rospy.get_param('~depth_topic', '/terrasentia/zed2/zed_node/depth/depth_registered')
        self.caminfo_topic = rospy.get_param('~camera_info_topic', '/terrasentia/zed2/zed_node/rgb/camera_info')
        self.thres = rospy.get_param('~obstacle_threshold', 0.0)
        self.dtype = torch.float16

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        rospy.loginfo(f'TRAVERSABILITY_MAP_NODE using device: {self.device}')

        params = Object()
        # Model parameters (input size 320x240)
        params.pretrained = False
        params.input_size = (320, 240)
        params.output_size = (320, 240)
        params.output_channels = 1
        params.bottleneck_dim = 256
        params.depth_mean = 1.295778
        params.depth_std = 3.441738

        self.input_size = params.input_size  
        self.output_size = (640, 480) 
        self.depth_mean = params.depth_mean
        self.depth_std = params.depth_std
        
        # Create network object
        self.net = TravNet(params).type(self.dtype).to(self.device)

        # Load trained weights
        rospy.loginfo(f'Loading model from {self.model_path}')
        if self.device.type == 'cuda':
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        self.net.load_state_dict(torch.load(self.model_path, map_location=map_location, weights_only=True))
        self.net.type(self.dtype)
        self.net.eval()

        # Publishers
        self.traversability_pub = rospy.Publisher('/terrasentia/traversability_map/compressed', CompressedImage, queue_size=1)
        self.caminfo_pub = rospy.Publisher('/terrasentia/traversability_map/camera_info', CameraInfo, queue_size=1)

        # Image filter subscribers
        rgb_sub = message_filters.Subscriber(self.image_topic, CompressedImage, queue_size=1, buff_size=2**24)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=2**24)
        caminfo_sub = message_filters.Subscriber(self.caminfo_topic, CameraInfo, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, caminfo_sub], 1, 0.3)
        ts.registerCallback(self.callback)

        rospy.spin()

    def callback(self, rgb_msg, depth_msg, caminfo_msg):
        # Prepare RGB image
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logerr("Failed to decode RGB image")
            return
        
        # Resize image to the model's input size (320x240)
        color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, self.input_size, interpolation=cv2.INTER_AREA)

        image_tensor = self.transform(color_img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image = image_tensor.type(self.dtype).to(self.device)

        # Prepare Depth image
        np_arr = np.frombuffer(depth_msg.data, np.float32)
        try:
            depth_img = np_arr.reshape(depth_msg.height, depth_msg.width)
        except ValueError:
            rospy.logerr("Depth image shape does not match expected dimensions")
            return

        # Resize depth image to 320x240 (same as input size)
        depth_img = cv2.resize(depth_img, self.input_size, interpolation=cv2.INTER_NEAREST)
        depth_mask = np.isfinite(depth_img)

        # Filter out the NaNs in depth image
        depth_img[~depth_mask] = 0.0

        # Transform depth data to tensor
        depth_img = (depth_img - self.depth_mean) / self.depth_std
        depth_img = np.stack((depth_img, depth_mask), axis=2)
        depth_img = np.transpose(depth_img, (2, 0, 1))
        depth_tensor = torch.from_numpy(depth_img)
        depth_tensor = depth_tensor.unsqueeze_(0)
        depth = depth_tensor.type(self.dtype).to(self.device)

        with torch.no_grad():           
            pred = self.net(image, depth)
        
        im = 255 * pred[0, 0, :, :].cpu().numpy()
        im = im.astype('uint8')

        im[im < self.thres * 255] = 0

        # Resize output image to 640x480 for publishing
        im_resized = cv2.resize(im, self.output_size, interpolation=cv2.INTER_LINEAR)
        
        _, im_encoded = cv2.imencode('.jpg', im_resized)
        msg = CompressedImage()
        msg.header = rgb_msg.header
        msg.format = "jpeg"
        msg.data = np.array(im_encoded).tobytes()

        # Publish compressed image
        self.traversability_pub.publish(msg)

        # Scale the CameraInfo message
        scaled_caminfo_msg = self.scale_camera_info(caminfo_msg, self.output_size[0], self.output_size[1])
        self.caminfo_pub.publish(scaled_caminfo_msg)

    def scale_camera_info(self, caminfo_msg, new_width, new_height):
        """Scale the CameraInfo message to match the resized output image dimensions."""
        scale_x = new_width / caminfo_msg.width
        scale_y = new_height / caminfo_msg.height

        scaled_caminfo_msg = CameraInfo()
        scaled_caminfo_msg.header = caminfo_msg.header
        scaled_caminfo_msg.width = new_width
        scaled_caminfo_msg.height = new_height

        # Scale the intrinsic camera matrix
        scaled_caminfo_msg.K = [
            caminfo_msg.K[0] * scale_x, caminfo_msg.K[1], caminfo_msg.K[2] * scale_x,
            caminfo_msg.K[3], caminfo_msg.K[4] * scale_y, caminfo_msg.K[5] * scale_y,
            caminfo_msg.K[6], caminfo_msg.K[7], caminfo_msg.K[8]
        ]

        # Scale the projection matrix
        scaled_caminfo_msg.P = [
            caminfo_msg.P[0] * scale_x, caminfo_msg.P[1], caminfo_msg.P[2] * scale_x, caminfo_msg.P[3] * scale_x,
            caminfo_msg.P[4] * scale_y, caminfo_msg.P[5] * scale_y, caminfo_msg.P[6] * scale_y, caminfo_msg.P[7] * scale_y,
            caminfo_msg.P[8], caminfo_msg.P[9], caminfo_msg.P[10], caminfo_msg.P[11]
        ]

        # Copy the rest of the fields without modification
        scaled_caminfo_msg.D = caminfo_msg.D
        scaled_caminfo_msg.distortion_model = caminfo_msg.distortion_model
        scaled_caminfo_msg.R = caminfo_msg.R

        return scaled_caminfo_msg

if __name__ == '__main__':
    TRAVERSABILITY_NODE()
