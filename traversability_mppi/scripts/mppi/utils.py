import rospy
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tf
import sensor_msgs.point_cloud2 as pc2

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

def visualize_rollouts(rollouts, costs, frame_id="map"):
    # Header for the PointCloud2 message
    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()

    # Create fields for the PointCloud2 message
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]
    
    reshaped_rollouts = rollouts.reshape(-1, 3)  # Flatten to shape (N_points, 3) where N_points = num_rollouts * num_points
    reshaped_costs = costs.flatten()  # Flatten costs into a 1D array

    # Normalize the costs to be between 0 and 1
    norm_costs = np.clip(reshaped_costs, 0, 1)

    # Generate colors using the jet colormap in one step
    rgba_colors = plt.cm.jet(norm_costs)

    # Convert RGBA to packed RGB format expected by PointCloud2
    packed_rgb = ((rgba_colors[:, 2] * 255).astype(np.uint32) |
                  (rgba_colors[:, 1] * 255).astype(np.uint32) << 8 |
                  (rgba_colors[:, 0] * 255).astype(np.uint32) << 16)

    # Z-coordinate is zero since we're visualizing in 2D
    z_coords = np.zeros_like(reshaped_rollouts[:, 0])

    # Create a structured array with x, y, z, and rgb fields
    cloud_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32)
    ])

    points = np.empty(reshaped_rollouts.shape[0], dtype=cloud_dtype)
    points['x'] = reshaped_rollouts[:, 0]
    points['y'] = reshaped_rollouts[:, 1]
    points['z'] = z_coords
    points['rgb'] = packed_rgb

    # Create the PointCloud2 message using numpy batch data
    pointcloud = pc2.create_cloud(header, fields, points)

    return pointcloud

def visualize_rollouts_intesity(rollouts, costs, frame_id="map"):
    # Header for the PointCloud2 message
    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()

    # Create fields for the PointCloud2 message with intensity
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    
    reshaped_rollouts = rollouts.reshape(-1, 3)  # Flatten to shape (N_points, 3)
    reshaped_costs = costs.flatten()  # Flatten costs into a 1D array

    # Normalize the costs to be between 0 and 1 for intensity values
    norm_costs = np.clip(reshaped_costs, 0, 1)

    # Z-coordinate is zero since we're visualizing in 2D
    z_coords = np.zeros_like(reshaped_rollouts[:, 0])

    # Create a structured array with x, y, z, and intensity fields
    cloud_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)
    ])

    points = np.empty(reshaped_rollouts.shape[0], dtype=cloud_dtype)
    points['x'] = reshaped_rollouts[:, 0]
    points['y'] = reshaped_rollouts[:, 1]
    points['z'] = z_coords
    points['intensity'] = norm_costs

    # Create the PointCloud2 message using numpy batch data
    pointcloud = pc2.create_cloud(header, fields, points)

    return pointcloud

def mppi_path_msg(prediction, frame_id='map'):
    # Create a Path message
    mppi_out = Path()
    mppi_out.header.frame_id = frame_id  # Set the frame for the Path message
    mppi_out.header.stamp = rospy.Time.now()  # Set the current time as the timestamp

    for i in range(prediction.shape[0]):
        # Create a PoseStamped message for each point in the prediction
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id  # Use the same frame
        pose_stamped.header.stamp = rospy.Time.now()

        # Set the position (x, y)
        pose_stamped.pose.position.x = prediction[i, 0]
        pose_stamped.pose.position.y = prediction[i, 1]
        pose_stamped.pose.position.z = 0.0  # Assuming 2D plane, z is 0

        # Convert the heading (yaw) to quaternion for the orientation
        yaw = prediction[i, 2]  # Assuming prediction[i, 2] contains the heading angle (yaw)
        quat = tf.quaternion_from_euler(0, 0, yaw)  # Convert yaw to quaternion
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        # Append the PoseStamped to the Path
        mppi_out.poses.append(pose_stamped)

    return mppi_out