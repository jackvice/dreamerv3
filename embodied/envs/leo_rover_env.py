"""
ROS2 Gym Environment for Leo Rover Vision-Based Navigation
---------------------------------------------------------
A custom Gym environment for training a Leo rover in vision-based navigation 
tasks using ROS2 and reinforcement learning, with an emphasis on functional 
programming style and absolute positioning for off-road scenarios.
"""

# Add to the top of your file with other imports
import csv
from datetime import datetime
import gym
import numpy as np
import rclpy
from rclpy.node import Node
import time
import math
import os
from typing import Dict, Tuple, List, Optional, Any, Callable
from geometry_msgs.msg import Twist, PoseArray, Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler
from gym import spaces
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
from collections import deque
import atexit
import subprocess
from datetime import datetime


# Global ROS2 initialization management
_ROS_INITIALIZED = False

def safe_ros_init() -> bool:
    """Initialize ROS2 safely, handling the case where it's already initialized.
    
    Returns:
        bool: True if this function initialized ROS2, False if it was already initialized
    """
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

def safe_ros_shutdown() -> None:
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
                 size: Tuple[int, int] = (96, 96),
                 length: int = 6000,
                 image_topic: str = '/leo1/camera/image_raw',
                 cmd_vel_topic: str = '/leo1/cmd_vel',
                 pose_topic: str = '/rover/pose_array',
                 goal_x_range: Tuple[float, float] = (0.0, 8.0),
                 goal_y_range: Tuple[float, float] = (1.0, 6.0),
                 max_linear_velocity: float = 0.5,
                 max_angular_velocity: float = 1.0,
                 connection_timeout: float = 10.0,
                 **kwargs: Any) -> None:
        """Initialize the environment with the given parameters.
        
        Args:
            size: Image size as (width, height)
            length: Maximum number of steps per episode
            image_topic: ROS topic for camera images
            cmd_vel_topic: ROS topic for velocity commands
            pose_topic: ROS topic for absolute positioning
            goal_x_range: Range for random goal x positions
            goal_y_range: Range for random goal y positions
            max_linear_velocity: Maximum linear velocity
            max_angular_velocity: Maximum angular velocity
            connection_timeout: Timeout for checking topic connections
            **kwargs: Additional parameters
        """
        super().__init__()
        
        # Environment parameters
        self.image_size = size
        self.max_episode_steps = length
        self.goal_x_range = goal_x_range
        self.goal_y_range = goal_y_range
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # Initialize ROS2 node - safely
        self.initialized_ros = safe_ros_init()
        self.node = rclpy.create_node(f'leo_rover_controller_{id(self)}')
        self.bridge = CvBridge()
        
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
        self.initial_image_saved = False


        now = datetime.now()
        self.timestamp_str = now.strftime("%Y%m%d%H%M")

        # Flags
        self.received_image = False
        self.received_pose = False
        
        # Stuck detection
        self.position_history = deque(maxlen=500)
        self.stuck_threshold = 0.3  # Minimum distance to move in 100 steps
        
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
        
        # Set up QoS profile for pose array topic (BEST_EFFORT to match the bridge)
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to absolute position topic (PoseArray) with appropriate QoS
        self.pose_subscriber = self.node.create_subscription(
            PoseArray,
            pose_topic,
            self.pose_callback,
            pose_qos  # Use the QoS profile that matches the publisher
        )
        
        # Wait for connections
        self.check_connections(timeout=connection_timeout)
        
        # Initialize target position
        self.set_new_target()


    def respawn_robot(self, x=0.0, y=0.0, z=2.0):
        """Respawn the robot at a specific position with random orientation.
        
        Args:
            x: X position to spawn at
            y: Y position to spawn at
            z: Z position to spawn at
        """
        # Generate random orientation (yaw)
        final_yaw = np.random.uniform(-np.pi, np.pi)
        
        # Convert to quaternion
        quat_w = np.cos(final_yaw / 2)
        quat_z = np.sin(final_yaw / 2)
        
        # Define the world and robot name
        world_name = 'leo_marsyard'  # Default to maze if not specified
        robot_name = 'leo_rover_leo1'  # Use your robot name here
        
        # Define the pose service path
        pose_service_path = f'/world/{world_name}/set_pose'
        
        print(f"Respawning robot at ({x:.2f}, {y:.2f}, {z:.2f}) with yaw {math.degrees(final_yaw):.1f}°")
        
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', pose_service_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', f'name: "{robot_name}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: 0, y: 0, z: {quat_z}, w: {quat_w}}}'
            ]
            result = subprocess.run(reset_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to reset robot pose: {result.stderr}")
        except Exception as e:
            print(f"Error executing reset command: {str(e)}")
        
        # Stop the robot
        self.stop_robot()
        
        # Wait for a moment to let the physics settle
        time.sleep(0.5)


        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one time step within the environment.
        
        Args:
            action: Array with [linear_velocity, angular_velocity]
            
        Returns:
            observation: Dict containing the current observation
            reward: Reward value
            done: Whether the episode is done
            info: Additional information
        """
        self.steps += 1
        self.total_steps += 1
        
        # Extract and apply actions
        linear_velocity = float(action[0])
        angular_velocity = float(action[1])

        # Update current velocity values (ADD THESE LINES)
        self.current_linear_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity
    
        # Create and publish command
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.cmd_vel_publisher.publish(twist)
        
        # Wait for sensor data
        self.spin_until_sensor_data()
        
        # Update position history for stuck detection
        self.position_history.append((self.current_position[0], self.current_position[1]))
        
        # Check if robot is stuck
        if len(self.position_history) >= self.position_history.maxlen:
            stuck, distance_moved = self.check_if_stuck()
            if stuck:
                print(f"Robot is stuck, has moved only {distance_moved:.3f} meters in {self.position_history.maxlen} steps")
                observation = self.get_observation()
                self.respawn_robot(0.0, 0.0, 2.5)
                self.done = True
                return observation, -10.0, True, {'status': 'stuck'}
        
        # Check if robot has flipped
        flipped = self.check_if_flipped()
        if flipped or self.current_position[2] < -10: # flipped or fell off world
            print("Robot has flipped, roll: ", self.current_roll, "pitch:", self.current_pitch)
            self.respawn_robot(0.0, 0.0, 1.5)
            observation = self.get_observation()
            self.done = True
            return observation, -30.0, True, {'status': 'flipped'}
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = (self.steps >= self.max_episode_steps)
        self.done = done
        
        # Check if goal reached
        distance_to_goal = self.get_distance_to_target()
        if distance_to_goal < 0.5:  # Goal threshold
            print(f"Goal reached! Distance: {distance_to_goal:.3f}")
            reward += 100.0  # Bonus reward for reaching goal
            self.set_new_target()  # Set a new target
        
        # Log info occasionally
        if self.total_steps % 1000 == 0:
            self.log_status()
        
        observation = self.get_observation()
        info = {
            'distance_to_goal': distance_to_goal,
            'steps': self.steps,
            'total_steps': self.total_steps
        }
        
        return observation, reward, done, info

   
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment to its initial state.
        
        Args:
            seed: Random seed
            
        Returns:
            observation: Dict containing the initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Stop the robot
        self.stop_robot()
        
        # Reset internal state
        self.steps = 0
        self.position_history.clear()
        self.done = False
        
        # Reset target position
        self.set_new_target()
        self.previous_distance = None
        
        # Wait for sensor data to update after reset
        time.sleep(0.5)
        self.spin_until_sensor_data()

        # Save the first image only once
        if not self.initial_image_saved and self.received_image:
            image_path = os.path.join(os.getcwd(), "obs_image.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
            print(f"Saved initial image to {image_path}")
            self.initial_image_saved = True  # Mark as saved
        
        return self.get_observation()
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop_robot()
        try:
            self.node.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return the current observation.
        
        Returns:
            Dict containing the observation
        """
        return {
            'image': self.current_image,
            'imu': np.array([self.current_roll, self.current_pitch, self.current_yaw], 
                           dtype=np.float32),
            'target': self.get_target_info(),
            'velocities': np.array([self.current_linear_velocity, self.current_angular_velocity],
                                  dtype=np.float32)
        }

    # Function to format values to three significant digits
    def format_value(self, value):
        return f"{value:.3g}"  # 3 significant digits

    def log_reward(self, relative_angle, distance_to_goal, prev_distance, distance_delta, 
                  distance_reward, time_penalty, speed_bonus, total_reward):
        """Log reward components to CSV file."""
        
        log_file = "/home/jack/logdir/" + self.timestamp_str + "_reward_logs.csv"
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total_steps", "heading_to_goal", "distance_to_goal", "prev_distance", 
                               "distance_delta", "distance_reward", "time_penalty", 
                               "speed_bonus", "total_reward"])
        relative_angle = math.degrees(relative_angle)  # Convert radians to degrees
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.format_value(self.total_steps),
                self.format_value(relative_angle),
                self.format_value(distance_to_goal),
                self.format_value(prev_distance),
                self.format_value(distance_delta),
                self.format_value(distance_reward),
                self.format_value(time_penalty),
                self.format_value(speed_bonus),
                self.format_value(total_reward)
            ])


    # Modified calculate_reward function
    def calculate_reward(self) -> float:
        """Calculate the reward based on current state."""
        # Get current distance to target
        current_distance = self.get_distance_to_target()
        
        # Initialize previous distance if needed
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Calculate distance delta (positive means getting closer)
        distance_delta = self.previous_distance - current_distance
        
        # Get target info
        target_info = self.get_target_info()
        relative_angle = target_info[1]
        
        # Base reward components
        distance_reward = distance_delta #* 2.0  # Scale factor for distance progress
        
        # Efficiency penalty (small penalty for time)
        time_penalty = -0.01
        
        # Speed bonus (encourage moving when making progress AND pointing toward target)
        speed_bonus = 0.0
        if distance_delta > 0 and abs(relative_angle) < math.pi/2:  # Within 90° of target direction
            speed_bonus = self.current_linear_velocity * 0.1
        
        # Combine reward components
        total_reward = distance_reward + time_penalty + speed_bonus
        
        # Log rewards every 1000 steps
        if self.total_steps % 1000 == 0:
            try:
                self.log_reward(
                    relative_angle = relative_angle,
                    distance_to_goal=current_distance,
                    prev_distance=self.previous_distance,
                    distance_delta=distance_delta,
                    distance_reward=distance_reward,
                    time_penalty=time_penalty,
                    speed_bonus=speed_bonus,
                    total_reward=total_reward
                )
            except Exception as e:
                print(f"Error logging reward components: {e}")
        
        # Store current distance for next step
        self.previous_distance = current_distance
        
        return total_reward
    
    def calculate_rewardold(self) -> float:
        """Calculate the reward based on current state.
        
        Returns:
            Calculated reward value
        """
        # Get current distance to target
        current_distance = self.get_distance_to_target()
        
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
        if distance_delta > 0 and abs(relative_angle) < math.pi/2:  # Within 90° of target direction
            speed_bonus = abs(self.current_linear_velocity) * 0.1
        
        # Combine reward components
        reward = distance_reward + time_penalty + speed_bonus
        
        # Store current distance for next step
        self.previous_distance = current_distance

        return reward
    
    def get_distance_to_target(self) -> float:
        """Calculate distance to current target.
        
        Returns:
            Distance to target in meters
        """
        return math.sqrt(
            (self.current_position[0] - self.target_x)**2 + 
            (self.current_position[1] - self.target_y)**2
        )
    
    def get_target_info(self) -> np.ndarray:
        """Calculate distance and relative angle to current target.
        
        Returns:
            Array containing [distance, relative_angle]
        """
        # Calculate distance
        distance = self.get_distance_to_target()
        
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
    
    def set_new_target(self) -> None:
        """Set a new random target position."""
        self.target_x = np.random.uniform(*self.goal_x_range)
        self.target_y = np.random.uniform(*self.goal_y_range)
        print(f"New target set at ({self.target_x:.2f}, {self.target_y:.2f})")
    
    def check_if_flipped(self) -> bool:
        """Check if the robot has flipped based on orientation data.
        
        Returns:
            True if the robot is flipped, False otherwise
        """
        FLIP_THRESHOLD = 1.7  # ~75 degrees in radians
        return abs(self.current_roll) > FLIP_THRESHOLD or abs(self.current_pitch) > FLIP_THRESHOLD
    
    def check_if_stuck(self) -> Tuple[bool, float]:
        """Check if the robot is stuck by analyzing position history.
        
        Returns:
            Tuple of (is_stuck, distance_moved)
        """
        if len(self.position_history) < self.position_history.maxlen:
            return False, 0.0
            
        start_pos = self.position_history[0]
        end_pos = self.position_history[-1]
        distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 + 
                                  (end_pos[1] - start_pos[1])**2)
        
        return distance_moved < self.stuck_threshold, distance_moved
    
    def stop_robot(self) -> None:
        """Send stop command to the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)
    
    def spin_until_sensor_data(self, timeout: float = 1.0) -> None:
        """Spin ROS node until we get fresh sensor data or timeout.
        
        Args:
            timeout: Maximum time to wait for sensor data
        """
        start_time = time.time()
        self.received_image = False
        self.received_pose = False
        
        while not (self.received_image and self.received_pose):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if time.time() - start_time > timeout:
                break
    
    def check_connections(self, timeout: float) -> None:
        """Check if all required topics are publishing data.
        
        Args:
            timeout: Maximum time to wait for connections
        """
        start_time = time.time()
        while not (self.received_image and self.received_pose):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                missing = []
                if not self.received_image:
                    missing.append("camera")
                if not self.received_pose:
                    missing.append("pose array")
                
                if missing:
                    self.node.get_logger().warn(f"Timed out waiting for {', '.join(missing)} data")
                break
    
    def log_status(self) -> None:
        """Log current status information."""
        print(f"Steps: {self.steps}/{self.max_episode_steps}, "
              f"Position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}), "
              f"Target: ({self.target_x:.2f}, {self.target_y:.2f}), "
              f"Distance: {self.get_distance_to_target():.2f}, "
              f"Velocity: linear={self.current_linear_velocity:.2f}, angular={self.current_angular_velocity:.2f}")
    
    # Callback functions for ROS topics
    def image_callback(self, msg: Image) -> None:
        """Process camera image data.
        
        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
            # Resize image to desired dimensions
            self.current_image = cv2.resize(cv_image, self.image_size, 
                                          interpolation=cv2.INTER_AREA)
            
            self.received_image = True
        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")
    
    def pose_callback(self, msg: PoseArray) -> None:
        """Process pose array data for absolute positioning.
        
        Args:
            msg: ROS PoseArray message
        """
        try:
            if len(msg.poses) > 0:
                # Get the first pose in the array (assuming it's the robot's pose)
                pose = msg.poses[0]
                
                # Extract position
                self.current_position[0] = pose.position.x
                self.current_position[1] = pose.position.y
                self.current_position[2] = pose.position.z
                
                # Extract orientation
                self.current_orientation[0] = pose.orientation.w
                self.current_orientation[1] = pose.orientation.x
                self.current_orientation[2] = pose.orientation.y
                self.current_orientation[3] = pose.orientation.z
                
                # Extract roll, pitch, yaw from quaternion
                quat = [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z
                ]
                self.current_roll, self.current_pitch, self.current_yaw = quat2euler(quat, axes='sxyz')
                
                self.received_pose = True
                
                # Debug output - remove after confirming it works
                if self.total_steps % 1000 == 0:
                    print(f"Received pose: pos=({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})")
            else:
                self.node.get_logger().warn("Received empty pose array")
        except Exception as e:
            self.node.get_logger().error(f"Error processing pose data: {e}")


# Register clean-up to ensure ROS is shut down properly
atexit.register(safe_ros_shutdown)
