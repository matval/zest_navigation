#!/usr/bin/env python3
import copy
import torch
import numpy as np

import rospy

from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import PointCloud2, Joy
from geometry_msgs.msg import TwistStamped, PoseStamped

from mppi.mppi import MPPI
from mppi.utils import visualize_rollouts, mppi_path_msg, generate_interpolated_reference

class TRAVERSABILITY_MPPI_NODE:
    def __init__(self):
        # Define ROS topics
        cmd_topic       = rospy.get_param('~cmd_topic', 'cmd_vel')
        odom_topic      = rospy.get_param('~odom_topic', 'odom')
        joy_topic       = rospy.get_param('~joy_topic', '/terrasentia/joy')
        # goal_topic      = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        reference_topic = rospy.get_param('~reference_topic', 'planned_path')
        map_topic       = rospy.get_param('~map_topic', 'grid_map')
        self.verbose    = rospy.get_param('~verbose', False)

        # Load configuration params
        self.N = rospy.get_param('~horizon', 20)
        self.dt = rospy.get_param('~delta_time', 0.2)
        num_samples = rospy.get_param('~num_samples', 1000)
        gamma = rospy.get_param('~gamma', 1.0)
        noise_mu = rospy.get_param('~noise_mu', [0.0, 0.0])
        noise_sigma = rospy.get_param('~noise_sigma', [0.4, 1.0])
        u_min = rospy.get_param('~u_min', [-1.0, -3.0])
        u_max = rospy.get_param('~u_max', [1.0, 3.0])
        self.reference_speed = rospy.get_param('~reference_speed', 0.6)
        robot_length = rospy.get_param('~robot_length', 0.5)
        robot_width = rospy.get_param('~robot_width', 0.3)

        self.run_mppi = False
        self.publish_rollouts = True
        self.previous_button_state = False
        
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('MPPI NODE using device:', self.device)

        self.odom_received = False
        self.trav_received = False
        self.ref_received = False

        # Create network object
        self.mppi = MPPI(
            dt          = self.dt,
            N           = self.N,
            K           = num_samples,
            gamma       = gamma,
            noise_mu    = noise_mu,
            noise_sigma = noise_sigma,
            u_min       = u_min,
            u_max       = u_max,
            Q           = [1.0, 1.0, 0.1],
            Qf          = [2.0, 2.0, 0.1],
            R           = [50.0, 0.01],
            W           = [200.0, 1.0],
            robot_length= robot_length,
            robot_width = robot_width
        ).to(self.device)
        self.mppi.eval()

        # Create empty reference trajectory
        self.mpc_reference = {
            'x': [],
            'y': [],
            'theta': [],
            'speed': np.array([]),
            'omega': np.array([])
        }

        # Subscribers
        rospy.Subscriber(joy_topic, Joy, self.joy_callback)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        # rospy.Subscriber(goal_topic, PoseStamped, self.reference_callback, queue_size=1)
        rospy.Subscriber(reference_topic, Path, self.reference_callback, queue_size=1)
        rospy.Subscriber(map_topic, GridMap, self.traversability_callback, queue_size=1, buff_size=2**24)

        # Publishers
        self.pub_cmd = rospy.Publisher(cmd_topic, TwistStamped, queue_size=1)
        self.pub_path = rospy.Publisher('mppi_path', Path, queue_size=1)
        self.pub_ref = rospy.Publisher('mppi_reference', Path, queue_size=1)
        # self.pub_traversability_image = rospy.Publisher('/mppi_traversability_image_debug', Image, queue_size=1)

        # Publish rollouts
        if self.publish_rollouts:
            self.pub_rollouts = rospy.Publisher('mppi_rollouts', PointCloud2, queue_size=1)

        # Set loop rate
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.run_navigation()
            rate.sleep()

    def run_navigation(self):
        # Run only when a new reference and map arrives
        if not self.ref_received or not self.trav_received or not self.odom_received:
            return

        odom = copy.deepcopy(self.current_pose)
        mpc_reference = self.update_reference()
        trav_map = copy.deepcopy(self.trav_map)
        grid_bounds = copy.deepcopy(self.grid_bounds)

        if len(mpc_reference['x']) == 0 or not self.run_mppi:
            if self.verbose:
                rospy.loginfo("TRAV-MPPI empty wps")
            
            mpc_cmd = TwistStamped()
            mpc_cmd.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(mpc_cmd)
        
        else:
            if self.verbose:
                rospy.loginfo("TRAV-MPPI wps are not empty")

            # Convert odometry to tensor
            odom_tensor = torch.tensor([
                odom['x'],
                odom['y'],
                odom['theta']], dtype=self.dtype, device=self.device).t().unsqueeze(0)

            # Move data to GPU
            reference = torch.tensor([
                mpc_reference['x'],
                mpc_reference['y'],
                mpc_reference['theta'],
                mpc_reference['v'],
                mpc_reference['omega']], dtype=self.dtype, device=self.device).t().unsqueeze(0)

            trav_map = torch.tensor(trav_map, dtype=self.dtype, device=self.device).unsqueeze(0)

            # And run MPPI
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    a_hat, trajs, pred_states, probs, travs = self.mppi(odom_tensor, reference, trav_map, grid_bounds)

            prediction = trajs[0].cpu().detach().numpy()
            rollouts = pred_states[0].cpu().detach().numpy()
            actions = a_hat[0].cpu().detach().numpy()
            probs = probs[0].cpu().detach().numpy()
            travs = travs[0].cpu().detach().numpy()

            # Publish command
            mpc_cmd = TwistStamped()
            mpc_cmd.header.stamp = rospy.Time.now()
            mpc_cmd.twist.linear.x = actions[0,0]
            mpc_cmd.twist.angular.z = actions[0,1]
            self.pub_cmd.publish(mpc_cmd)

            if self.publish_rollouts:
                rollout_msg = visualize_rollouts(rollouts, travs[...,0], odom['z'], frame_id="odom")
                self.pub_rollouts.publish(rollout_msg)

            # Publish MPPI predicted output
            mppi_out = mppi_path_msg(prediction, odom['z'], frame_id="odom")
            # Publish outputs
            self.pub_path.publish(mppi_out)

    def update_reference(self):
        path = copy.deepcopy(self.reference_path)

        # Find the closest point on the path to the current position
        distances = [np.hypot(px - self.current_pose['x'], py - self.current_pose['y']) for px, py in path]
        closest_index = np.argmin(distances)

        if distances[-1] < 1.0:
            if self.verbose:
                print("Goal reached!")

            mpc_reference = {
                'x': [],
                'y': [],
                'theta': [0.0],
                'v': [0.0],
                'omega': [0.0]
            }

            return mpc_reference

        # Generate interpolated reference points
        horizon_length = self.N * self.reference_speed * self.dt
        x_ref_seq, y_ref_seq = generate_interpolated_reference(closest_index, path, horizon_length, self.N)

        # Clear previous references and set new ones
        mpc_reference = {}
        mpc_reference['x'] = x_ref_seq
        mpc_reference['y'] = y_ref_seq

        # Calculate heading for each point
        mpc_reference['theta'] = []
        for t in range(self.N):
            if t < self.N - 1:
                dx = x_ref_seq[t + 1] - x_ref_seq[t]
                dy = y_ref_seq[t + 1] - y_ref_seq[t]
            else:
                dx = x_ref_seq[t] - x_ref_seq[t - 1]
                dy = y_ref_seq[t] - y_ref_seq[t - 1]

            heading_ref = np.arctan2(dy, dx)

            # Check if speed is negative and invert the heading to face forward
            if self.reference_speed < 0:
                heading_ref = (heading_ref + np.pi) % (2 * np.pi) - np.pi

            mpc_reference['theta'].append(heading_ref)

        # Set constant velocity and zero angular velocity for simplicity
        mpc_reference['v'] = (np.ones(self.N) * self.reference_speed).tolist()
        mpc_reference['omega'] = np.zeros(self.N).tolist()

        # Publish reference path for debugging
        ref_msg = Path()
        ref_msg.header.frame_id = 'odom'
        ref_msg.header.stamp = rospy.Time.now()
        for x, y in zip(x_ref_seq, y_ref_seq):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            ref_msg.poses.append(pose)

        self.pub_ref.publish(ref_msg)

        return mpc_reference

    def odom_callback(self, msg):
        # Get yaw from quaternion
        q = msg.pose.pose.orientation
        theta = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2 + q.z**2))

        # Store current pose
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'theta': theta
        }

        self.odom_received = True

    def traversability_callback(self, msg):
        # Convert traversability grid map to numpy array
        index = msg.layers.index('traversability')
        # Get the corresponding matrix data from the grid_map
        data = msg.data[index].data
        nrows = int(msg.info.length_y / msg.info.resolution)
        ncols = int(msg.info.length_x / msg.info.resolution)
        # Convert to NumPy array
        traversability = np.array(data, dtype=np.float32).reshape((nrows, ncols))

        # Convert epistemic uncertainty grid map to numpy array
        index = msg.layers.index('lambda')
        # Get the corresponding matrix data from the grid_map
        data = msg.data[index].data
        # Convert to NumPy array
        epistemic = np.array(data, dtype=np.float32).reshape((nrows, ncols))

        # Store the traversability map
        self.trav_map = np.stack([traversability, epistemic], 0)

        min_x = msg.info.pose.position.x - msg.info.length_x / 2
        min_y = msg.info.pose.position.y - msg.info.length_y / 2
        max_x = msg.info.pose.position.x + msg.info.length_x / 2
        max_y = msg.info.pose.position.y + msg.info.length_y / 2

        self.grid_bounds = {
            'xbound': [min_x, max_x, msg.info.resolution],
            'ybound': [min_y, max_y, msg.info.resolution]
        }

        self.trav_received = True

        # # Convert the NumPy image to a ROS Image message using cv_bridge
        # bridge = CvBridge()
        # trav_img = (traversability * 255).astype(np.uint8)
        # ros_image = bridge.cv2_to_imgmsg(trav_img, encoding="mono8")  # Mono 8-bit encoding

        # # Publish the image
        # self.pub_traversability_image.publish(ros_image)

    def reference_callback(self, msg):
        # Prevent MPC from running while preparing the reference
        self.ref_received = False

        if self.verbose:
            print("Using reference as a sequence of points tracking.")

        # Store the path
        self.reference_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

        # Allow MPC to run
        self.ref_received = True

    ## Single point tracking
    # def reference_callback(self, msg):
    #     # First, let's prevent the MPC to run while we are preparing the reference
    #     self.ref_received = False

    #     if self.verbose:
    #         print("Using reference as a single point tracking.")

    #     self.mpc_reference['x'].clear()
    #     self.mpc_reference['y'].clear()
    #     self.mpc_reference['theta'].clear()
    #     self.mpc_reference['v'] = np.ones(self.N) * self.reference_speed
    #     self.mpc_reference['omega'] = np.zeros(self.N)
        
    #     for t in range(self.N):
    #         x_ref = msg.pose.position.x
    #         y_ref = msg.pose.position.y
    #         heading_ref = np.arctan2(y_ref, x_ref)

    #         # Check if speed is negative and invert the heading to face forward
    #         if self.mpc_reference['v'][t] < 0:
    #             # Normalize angle to [-pi,pi)
    #             heading_ref = (heading_ref + np.pi) % (2*np.pi) - np.pi

    #         self.mpc_reference['x'].append(x_ref)
    #         self.mpc_reference['y'].append(y_ref)
    #         self.mpc_reference['theta'].append(heading_ref)

    #     # Now MPC can run
    #     self.ref_received = True

    def joy_callback(self, msg):
        # Check if the combination L2 + B is pressed
        if msg.buttons[8] and msg.buttons[1] and not self.previous_button_state:
            self.run_mppi = not self.run_mppi

        # Update the previous button state
        self.previous_button_state = msg.buttons[2]

if __name__ == '__main__':
    rospy.init_node('traversability_mppi')
    TRAVERSABILITY_MPPI_NODE()