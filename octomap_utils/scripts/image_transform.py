#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class RgbToGrayscale:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/terrasentia/front_mono/image_rect_color/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher("/terrasentia/front_mono/image_rect_gray/compressed", CompressedImage, queue_size=1)

    def image_callback(self, msg):
        try:
            # Converte a mensagem ROS CompressedImage para uma imagem OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Converte a imagem RGB para grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # gray_image = np.ones_like(cv_image) * 255

        try:
            # Converte a imagem OpenCV de volta para uma mensagem ROS CompressedImage
            _, gray_img_encoded = cv2.imencode('.jpg', gray_image)
            gray_msg = CompressedImage()
            gray_msg.header = msg.header
            gray_msg.format = "jpeg"
            gray_msg.data = np.array(gray_img_encoded).tobytes()
            self.image_pub.publish(gray_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    rospy.init_node('rgb_to_grayscale_node', anonymous=True)
    rgb_to_grayscale = RgbToGrayscale()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
