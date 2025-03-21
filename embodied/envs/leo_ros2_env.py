# leo_ros2_env.py
import gym
import numpy as np
import rclpy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from gym import spaces
import time

class LeoROS2Env(gym.Env):
    """OpenAI Gym environment for Leo Rover in ROS2 Gazebo simulation"""
    #metadata = {'render_modes': ['human']}
    
    def __init__(self, image_size=(84, 84), robot_ns='leo1'):
        super().__init__()
        
        # Initialize ROS2 node
        rclpy.init()
        self.node = rclpy.create_node('leo_dreamer_controller')
        self.bridge = CvBridge()
        self.robot_ns = robot_ns
        self.image_size = image_size
        
        # Latest data containers
        self.latest_image = np.zeros((*image_size, 3), dtype=np.uint8)
        self.latest_imu = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self.image_received = False
        self.imu_received = False
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.node.create_publisher(
            Twist,
            f'/{robot_ns}/cmd_vel',
            10
        )
        
        self.image_sub = self.node.create_subscription(
            Image,
            f'/{robot_ns}/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.node.create_subscription(
            Imu,
            f'/{robot_ns}/imu/data_raw',
            self.imu_callback,
            10
        )
        
        # Define action and observation spaces
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),   # [min_lin_vel, min_ang_vel]
            high=np.array([1.0, 1.0]),    # [max_lin_vel, max_ang_vel]
            dtype=np.float32
        )
        
        # Observation space: camera image and IMU data
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=(*image_size, 3), 
                dtype=np.uint8
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            )
        })
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Wait for the first observations
        self._wait_for_observations()
    
    def image_callback(self, msg):
        """Process camera image from ROS2"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            # Resize to desired dimensions
            self.latest_image = cv2.resize(cv_image, self.image_size)
            self.image_received = True
        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
    
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
    
        return roll, pitch, yaw

    def imu_callback(self, msg):
        """Process IMU data from ROS2"""
        try:
            # Extract orientation as Euler angles (roll, pitch, yaw)
            quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            roll, pitch, yaw = self.quaternion_to_euler(
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            
            self.latest_imu = np.array([roll, pitch, yaw], dtype=np.float32)
            self.imu_received = True
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")
    
    def _wait_for_observations(self):
        """Wait until we receive initial observations"""
        timeout = 10.0  # seconds
        start_time = time.time()
        
        while not (self.image_received and self.imu_received):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.node.get_logger().warn("Timed out waiting for observations")
                break
    
    def step(self, action):
        """Execute one environment step"""
        # Process ROS2 messages
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # Send action to the robot
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)
        
        # Small delay to allow simulation to advance
        time.sleep(0.01)
        
        # Process any pending messages
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # Get observation
        observation = {
            'image': self.latest_image,
            'imu': self.latest_imu
        }
        
        # Calculate reward for forward movement
        reward = max(0.0, action[0])  # Positive reward for forward motion
        
        # Update step count and check for episode end
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps
        
        # Additional info
        info = {}
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Stop the robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # Reset step counter
        self.current_step = 0
        
        # Process any pending messages
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # Get initial observation
        observation = {
            'image': self.latest_image,
            'imu': self.latest_imu
        }
        
        return observation, {}
    
    def close(self):
        """Clean up resources"""
        # Stop the robot
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        
        # Cleanup ROS2 node
        self.node.destroy_node()
        rclpy.shutdown()
