import rospy
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tf
import sensor_msgs.point_cloud2 as pc2
from scipy.interpolate import interp1d

from nav_msgs.msg import Path
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from geometry_msgs.msg import PoseStamped

def visualize_rollouts(rollouts, costs, height, frame_id="map"):
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
    z_coords = np.ones_like(reshaped_rollouts[:, 0]) * height

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

def mppi_path_msg(prediction, height, frame_id='map'):
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
        pose_stamped.pose.position.z = height

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

def generate_interpolated_reference(closest_index, path, horizon_length, N):
    # Get accumulated distance along the path
    accumulated_distance = [0.0]
    for i in range(1, len(path)):
        point_a = path[i - 1]
        point_b = path[i]
        segment_length = np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        accumulated_distance.append(accumulated_distance[-1] + segment_length)

    # Parameterize the path by the accumulated distance
    x_original = [pt[0] for pt in path]
    y_original = [pt[1] for pt in path]

    # Create interpolation functions
    interp_func_x = interp1d(accumulated_distance, x_original, kind='cubic', fill_value='extrapolate')
    interp_func_y = interp1d(accumulated_distance, y_original, kind='cubic', fill_value='extrapolate')

    # Generate new parameter values for the MPC horizon
    t_new = np.linspace(accumulated_distance[closest_index], accumulated_distance[closest_index] + horizon_length, N)

    # Interpolated x and y values
    new_x = interp_func_x(t_new)
    new_y = interp_func_y(t_new)

    return new_x, new_y

def calculate_lookahead_point(closest_point, path, lookahead_distance):
    # Extract two-dimensional coordinates of the closest point
    closest_index = path.index(closest_point)
    current_point = closest_point

    # Initialize total_travelled and next_point
    total_travelled = 0.0
    next_point = None

    # Loop through the path to find the next point based on the lookahead distance
    for i in range(closest_index, len(path) - 1):
        point_a = path[i]
        point_b = path[i + 1]

        # Calculate the distance between point_a and point_b
        segment_length = ((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)**0.5

        if total_travelled + segment_length >= lookahead_distance:
            # Interpolate to find the exact lookahead point on the segment
            remaining_distance = lookahead_distance - total_travelled
            ratio = remaining_distance / segment_length
            lookahead_x = point_a[0] + ratio * (point_b[0] - point_a[0])
            lookahead_y = point_a[1] + ratio * (point_b[1] - point_a[1])
            next_point = (lookahead_x, lookahead_y)
            break
        total_travelled += segment_length

    return next_point

