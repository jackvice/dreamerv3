"""
ROS2 Gym Environment for Leo Rover Vision-Based Navigation
---------------------------------------------------------
A custom Gym environment for training a Leo rover in vision-based navigation 
tasks using ROS2 and reinforcement learning.
"""

import gym
import numpy as np
import rclpy
import subprocess
import time
import math
import os
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler
from gym import spaces
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
from collections import deque


# Global ROS2 initialization management
_ROS_INITIALIZED = False

def safe_ros_init():
    """Initialize ROS2 safely, handling the case where it's already initialized."""
    global _ROS_INITIALIZED
    if not _ROS_INITIALIZED:
        try:
            rclpy.init()
            _ROS_INITIALIZED = True
            return True
        except RuntimeError:
            # ROS is already initialized elsewhere
            _ROS_INITIALIZED = True
            return False
    return False

def safe_ros_shutdown():
    """Shut down ROS2 safely, only if we initialized it."""
    global _ROS_INITIALIZED
    if _ROS_INITIALIZED:
        try:
            rclpy.shutdown()
            _ROS_INITIALIZED = False
        except Exception as e:
            print(f"Error shutting down ROS2: {e}")


class LeoRoverEnv(gym.Env):
    """Custom Gym environment for the Leo Rover with vision-based navigation"""
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 # Accept 'size' parameter to match DreamerV3 convention
                 size=(96, 96),
                 length=2000,
                 image_topic='/leo1/camera/image_raw',
                 imu_topic='/leo1/imu/data_raw',
                 cmd_vel_topic='/leo1/cmd_vel',
                 odom_topic='/leo1/odom',
                 goal_x_range=(-4.0, 4.0),
                 goal_y_range=(-4.0, 4.0),
                 init_x_range=(-4.0, 4.0),
                 init_y_range=(-4.0, 4.0),
                 max_linear_velocity=0.5,
                 max_angular_velocity=1.0,
                 connection_timeout=30,
                 **kwargs):  # Accept any other parameters from config
        
        super().__init__()
        
        # Map DreamerV3 parameter names to our internal names
        self.image_size = size  # Use 'size' instead of 'image_size'
        self.max_episode_steps = length  # Use 'length' instead of 'max_episode_steps'
        
        # Initialize ROS2 node - safely
        self._initialized_ros = safe_ros_init()
        self.node = rclpy.create_node(f'leo_rover_controller_{id(self)}')  # Use unique node name
        self.bridge = CvBridge()
        
        # Environment parameters
        self.goal_x_range = goal_x_range
        self.goal_y_range = goal_y_range
        self.init_x_range = init_x_range
        self.init_y_range = init_y_range
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # State variables
        self.steps = 0
        self.total_steps = 0
        self.current_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        self.current_position = np.zeros(3, dtype=np.float32)
        self.current_orientation = np.zeros(4, dtype=np.float32)  # quaternion
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.previous_distance = None
        self.done = False
        
        # Flags
        self.received_image = False
        self.received_imu = False
        self.received_odom = False
        
        # Stuck detection
        self.position_history = deque(maxlen=100)
        self.stuck_threshold = 0.05  # Minimum distance to move in 100 steps
        
        # Define action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-max_linear_velocity, -max_angular_velocity]),
            high=np.array([max_linear_velocity, max_angular_velocity]),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=(*self.image_size, 3), 
                dtype=np.uint8
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'target': spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([100, np.pi]),
                shape=(2,),
                dtype=np.float32
            ),
            'velocities': spaces.Box(
                low=np.array([-max_linear_velocity, -max_angular_velocity]),
                high=np.array([max_linear_velocity, max_angular_velocity]),
                shape=(2,),
                dtype=np.float32
            ),
        })
        
        # Initialize publishers and subscribers
        self.cmd_vel_publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )
        
        self.image_subscriber = self.node.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10
        )
        
        self.odom_subscriber = self.node.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Wait for connections
        self._check_connections(timeout=connection_timeout)
        
        # Initialize target position
        self._set_new_target()
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.steps += 1
        self.total_steps += 1
        
        # Extract and apply actions
        linear_velocity = float(action[0])
        angular_velocity = float(action[1])
        
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.cmd_vel_publisher.publish(twist)
        
        # Wait for sensor data
        self._spin_until_sensor_data()
        
        # Update position history for stuck detection
        self.position_history.append((self.current_position[0], self.current_position[1]))
        
        # Check if robot is stuck
        if len(self.position_history) >= self.position_history.maxlen:
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 + 
                                      (end_pos[1] - start_pos[1])**2)
            
            if distance_moved < self.stuck_threshold:
                print(f"Robot is stuck, has moved only {distance_moved:.3f} meters in {self.position_history.maxlen} steps")
                observation = self.get_observation()
                self.done = True
                return observation, -10.0, True, {'status': 'stuck'}
        
        # Check if robot has flipped
        if self._is_robot_flipped():
            print("Robot has flipped")
            observation = self.get_observation()
            self.done = True
            return observation, -10.0, True, {'status': 'flipped'}
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (self.steps >= self.max_episode_steps)
        self.done = done
        
        # Check if goal reached
        distance_to_goal = self._get_distance_to_target()
        if distance_to_goal < 0.5:  # Goal threshold
            print(f"Goal reached! Distance: {distance_to_goal:.3f}")
            reward += 50.0  # Bonus reward for reaching goal
            self._set_new_target()  # Set a new target
        
        # Log info occasionally
        if self.total_steps % 500 == 0:
            self._log_status()
        
        observation = self.get_observation()
        info = {
            'distance_to_goal': distance_to_goal,
            'steps': self.steps,
            'total_steps': self.total_steps
        }
        
        return observation, reward, done, info
    
    def reset(self, seed=None):
        """Reset the environment to its initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        # Stop the robot
        self._stop_robot()
        
        # Reset internal state
        self.steps = 0
        self.position_history.clear()
        self.done = False
        
        # Reset robot position (this would require a service call in a real system)
        # For now, we'll just simulate a position reset and wait for sensor updates
        self._set_new_target()
        self.previous_distance = None
        
        # Wait for sensor data to update after reset
        time.sleep(0.5)
        self._spin_until_sensor_data()
        
        return self.get_observation()
    
    def close(self):
        """Clean up resources"""
        self._stop_robot()
        try:
            self.node.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")
    
    def get_observation(self):
        """Return the current observation"""
        return {
            'image': self.current_image,
            'imu': np.array([self.current_roll, self.current_pitch, self.current_yaw], 
                           dtype=np.float32),
            'target': self._get_target_info(),
            'velocities': np.array([self.current_linear_velocity, self.current_angular_velocity],
                                  dtype=np.float32)
        }
    
    def _calculate_reward(self):
        """Calculate the reward based on current state"""
        # Get current distance to target
        current_distance = self._get_distance_to_target()
        
        # Initialize previous distance if needed
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Calculate distance delta (positive means getting closer)
        distance_delta = self.previous_distance - current_distance
        
        # Base reward components
        distance_reward = distance_delta * 5.0  # Scale factor for distance progress
        
        # Efficiency penalty (small penalty for time)
        time_penalty = -0.01
        
        # Speed bonus (encourage moving when making progress)
        speed_bonus = 0.0
        if distance_delta > 0:  # If we're making progress
            speed_bonus = abs(self.current_linear_velocity) * 0.1
        
        # Combine reward components
        reward = distance_reward + time_penalty + speed_bonus
        
        # Store current distance for next step
        self.previous_distance = current_distance
        
        return reward
    
    def _get_distance_to_target(self):
        """Calculate distance to current target"""
        return math.sqrt(
            (self.current_position[0] - self.target_x)**2 + 
            (self.current_position[1] - self.target_y)**2
        )
    
    def _get_target_info(self):
        """Calculate distance and relative angle to current target"""
        # Calculate distance
        distance = self._get_distance_to_target()
        
        # Calculate relative angle
        target_heading = math.atan2(
            self.target_y - self.current_position[1],
            self.target_x - self.current_position[0]
        )
        
        relative_angle = math.atan2(
            math.sin(target_heading - self.current_yaw),
            math.cos(target_heading - self.current_yaw)
        )
        
        return np.array([distance, relative_angle], dtype=np.float32)
    
    def _set_new_target(self):
        """Set a new random target position"""
        self.target_x = np.random.uniform(*self.goal_x_range)
        self.target_y = np.random.uniform(*self.goal_y_range)
        print(f"New target set at ({self.target_x:.2f}, {self.target_y:.2f})")
    
    def _is_robot_flipped(self):
        """Check if the robot has flipped based on IMU data"""
        FLIP_THRESHOLD = 1.3  # ~75 degrees in radians
        return abs(self.current_roll) > FLIP_THRESHOLD or abs(self.current_pitch) > FLIP_THRESHOLD
    
    def _stop_robot(self):
        """Send stop command to the robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)
    
    def _spin_until_sensor_data(self, timeout=1.0):
        """Spin ROS node until we get fresh sensor data or timeout"""
        start_time = time.time()
        self.received_image = False
        self.received_imu = False
        self.received_odom = False
        
        while (not self.received_image or not self.received_imu or not self.received_odom):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if time.time() - start_time > timeout:
                break
    
    def _check_connections(self, timeout):
        """Check if all required topics are publishing data"""
        start_time = time.time()
        while (not self.received_image or not self.received_imu or not self.received_odom):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                missing = []
                if not self.received_image:
                    missing.append("camera")
                if not self.received_imu:
                    missing.append("IMU")
                if not self.received_odom:
                    missing.append("odometry")
                self.node.get_logger().warn(f"Timed out waiting for {', '.join(missing)} data")
                break
    
    def _log_status(self):
        """Log current status information"""
        print(f"Steps: {self.steps}/{self.max_episode_steps}, "
              f"Position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}), "
              f"Target: ({self.target_x:.2f}, {self.target_y:.2f}), "
              f"Distance: {self._get_distance_to_target():.2f}, "
              f"Velocity: linear={self.current_linear_velocity:.2f}, angular={self.current_angular_velocity:.2f}")
    
    # Callback functions for ROS topics
    def image_callback(self, msg):
        """Process camera image data"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
            # Resize image to desired dimensions
            self.current_image = cv2.resize(cv_image, self.image_size, 
                                          interpolation=cv2.INTER_AREA)
            
            self.received_image = True
        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")
    
    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            # Extract orientation quaternion
            quat = [
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            ]
            
            # Normalize quaternion if needed
            norm = np.linalg.norm(quat)
            if norm > 0:
                quat = [q / norm for q in quat]
                
                # Convert to Euler angles
                self.current_roll, self.current_pitch, self.current_yaw = quat2euler(quat, axes='sxyz')
                self.received_imu = True
            else:
                self.node.get_logger().warn("Received zero quaternion from IMU")
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")
    
    def odom_callback(self, msg):
        """Process odometry data"""
        try:
            # Extract position
            self.current_position[0] = msg.pose.pose.position.x
            self.current_position[1] = msg.pose.pose.position.y
            self.current_position[2] = msg.pose.pose.position.z
            
            # Extract orientation
            self.current_orientation[0] = msg.pose.pose.orientation.w
            self.current_orientation[1] = msg.pose.pose.orientation.x
            self.current_orientation[2] = msg.pose.pose.orientation.y
            self.current_orientation[3] = msg.pose.pose.orientation.z
            
            # Extract velocities
            self.current_linear_velocity = msg.twist.twist.linear.x
            self.current_angular_velocity = msg.twist.twist.angular.z
            
            self.received_odom = True
        except Exception as e:
            self.node.get_logger().error(f"Error processing odometry data: {e}")


# Register clean-up to ensure ROS is shut down properly
import atexit
atexit.register(safe_ros_shutdown)
