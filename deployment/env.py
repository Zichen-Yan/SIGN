import gym
import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, Quaternion
from mavros_msgs.msg import State, RCIn, PositionTarget
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import math
import torch
from PIL import Image as Img
from deployment.utils import ResizeCenterCropper
import lpips
import matplotlib.pyplot as plt
# from controller_msgs.msg import FlatTarget
import tf

class Env:
    def __init__(self, config, add_control, add_lpips, fix_obs, take_snapshot):
        super(Env, self).__init__()
        self.config = config
        self.add_control = add_control
        self.add_lpips = add_lpips
        self.fix_obs = fix_obs
        self.take_snapshot = take_snapshot
        self.done_flag = False

        action_cfg = config.habitat.task.actions.continuous_control

        self.min_lin_vel, self.max_lin_vel = action_cfg.lin_vel_range
        self.min_ang_vel, self.max_ang_vel = action_cfg.ang_vel_range

        self.min_abs_lin_speed = action_cfg.min_abs_lin_speed
        self.min_abs_ang_speed = action_cfg.min_abs_ang_speed

        factor = 1
        self.max_lin_vel = factor * self.max_lin_vel
        self.min_abs_lin_speed = factor * self.min_abs_lin_speed
        self.min_abs_ang_speed = factor * self.min_abs_ang_speed
        # ================================
        self.observation_spaces = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "imagegoal_sensor_v2": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            # "depth": gym.spaces.Box(low=0.0, high=1.0, shape=(480, 640, 1), dtype=np.float32)
        })
        self.action_spaces = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # ROS =============================
        rospy.init_node("uav_control_and_vision")

        self.depth_max_dis = 10.0
        self.current_state = State()
        self.last_state = State()
        self.current_local_position = PoseStamped()
        self.current_rc_in = RCIn()
        self.offboard_channel_value = 2000
        self.rgb_image = None
        self.depth_image = None
        self.current_rc_in = None
        self.bridge = CvBridge()

        # Control ===========
        if add_control:
            rospy.Subscriber("/mavros/state", State, callback=self.state_cb)
            rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_position_callback)
            rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_in_callback)
            # Publisher
            self.vel_pub = rospy.Publisher("/reference/flatsetpoint", FlatTarget, queue_size=10)
            self.last_req = rospy.Time.now()
        # ===================
        self.current_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_goal = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_depth = np.zeros((480, 640, 1), dtype=np.float32)

        self.obs_transform = ResizeCenterCropper(size=(128, 128))
        self.raw_current_goal = np.array(Img.open("deployment/target_image/target1.png"))
        self.current_goal = cv2.resize(self.raw_current_goal, (128, 128), interpolation=cv2.INTER_AREA)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rgb_node = RGBSubscriber()
        self.depth_node = DepthSubscriber(depth_max_dis=self.depth_max_dis)
        if add_lpips:
            self.loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)

    def state_cb(self, msg):
        self.last_state = self.current_state
        self.current_state = msg

    def local_position_callback(self, data):
        self.current_local_position = data

    def rc_in_callback(self, data):
        self.current_rc_in = data
        self.offboard_channel_value = self.current_rc_in.channels[6]  # Switch A: offboard channel

    def com_lpips(self, img1, img2):
        img1 = torch.from_numpy(img1).unsqueeze(0)
        img1 = img1.permute(0, 3, 1, 2)
        img1 = self.obs_transform(img1, None)

        img2 = torch.from_numpy(img2).unsqueeze(0)
        img2 = img2.permute(0, 3, 1, 2)
        img2 = self.obs_transform(img2, None)

        im_1 = (img1 / 255 - 0.5) * 2
        im_2 = (img2 / 255 - 0.5) * 2
        d = self.loss_fn(im_1.to(self.device), im_2.to(self.device))
        return d

    def pub_velocity(self, velocity_x, velocity_y, velocity_z, yaw_rate):
        target_velocity = FlatTarget()
        target_velocity.header = Header()
        target_velocity.header.stamp = rospy.Time.now()
        target_velocity.header.frame_id = "base_footprint"

        target_velocity.type_mask = FlatTarget.IGNORE_SNAP | FlatTarget.IGNORE_ACC | FlatTarget.IGNORE_JERK | \
                                    FlatTarget.IGNORE_POS | FlatTarget.IGNORE_YAW
        target_velocity.velocity.x = velocity_x
        target_velocity.velocity.y = velocity_y
        target_velocity.velocity.z = velocity_z
        target_velocity.yaw_dot = yaw_rate
        self.vel_pub.publish(target_velocity)

    def normalize_action(self, angular_velocity, linear_velocity):
        done = False
        linear_velocity = (linear_velocity + 1.0) / 2.0
        angular_velocity = (angular_velocity + 1.0) / 2.0

        # Scale actions
        linear_velocity = self.min_lin_vel + linear_velocity * (self.max_lin_vel - self.min_lin_vel)
        angular_velocity = self.min_ang_vel + angular_velocity * (self.max_ang_vel - self.min_ang_vel)

        if ((abs(linear_velocity) < self.min_abs_lin_speed and abs(angular_velocity) < self.min_abs_ang_speed) and
                not self.done_flag):
            print('finished by agent')
            done = True

        angular_velocity = np.deg2rad(angular_velocity)
        return angular_velocity, linear_velocity, done

    def step(self, action):
        ang, lin, done = self.normalize_action(action[0][0], action[0][1])
        target_z = 0.7 # 1.2
        vel_z = (self.current_local_position.pose.position.z - target_z) * (-1)
        vel_z = np.clip(vel_z, -0.2, 0.2)

        orientation = self.current_local_position.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

        position = self.current_local_position.pose.position
        pos = np.array([position.x, position.y, position.z])
        ori = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        if self.add_lpips and not self.done_flag:
            score = self.com_lpips(self.rgb_image, self.current_goal)
            if score < 0.38:
                print('finished by lpips')
                done = True

        if done:
            self.done_flag = True

        vel_x = np.cos(yaw - np.pi / 2.0) * lin
        vel_y = np.sin(yaw - np.pi / 2.0) * lin

        if self.add_control:
            if self.done_flag:
                self.pub_velocity(0.0, 0.0, vel_z, 0.0)
            else:
                self.pub_velocity(vel_x, vel_y, vel_z, ang)

        if self.take_snapshot:
            fig, ax = plt.subplots()
            ax.imshow(self.raw_rgb_image)
            fig.canvas.mpl_connect("key_press_event", self.on_key_press)
            plt.show(block=True)

        if self.fix_obs:
            observation = {"rgb": self.current_rgb, "depth": self.current_depth,
                           "imagegoal_sensor_v2": self.current_goal}
        else:
            self.get_rgb()
            self.get_depth()
            observation = {"rgb": self.rgb_image, "depth": self.depth_image, "imagegoal_sensor_v2": self.current_goal}
        return [(observation, self.done_flag, pos, ori)]

    def on_key_press(self, event):
        if event.key == "s":
            filename = "deployment/target_image/target9.png"
            img = Img.fromarray(self.raw_rgb_image)
            img.save(filename)
            print(f"Image saved to {filename}")

    def reset(self):
        if self.fix_obs:
            return [{"rgb": self.current_rgb, "depth": self.current_depth, "imagegoal_sensor_v2": self.current_goal}]
        else:
            self.get_depth()
            self.get_rgb()
            return [{"rgb": self.rgb_image, "depth": self.depth_image, "imagegoal_sensor_v2": self.current_goal}]

    def get_rgb(self):
        self.raw_rgb_image = self.rgb_node.get_latest_image()
        self.rgb_image = cv2.resize(self.raw_rgb_image, (128, 128), interpolation=cv2.INTER_AREA)

        if self.rgb_image is None:
            self.rgb_image = self.current_rgb
        # plt.imshow(self.rgb_image)
        # plt.show()

    def get_depth(self):
        depth_image = self.depth_node.get_latest_image()
        # plt.imshow(depth_image, cmap="gray", vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        if depth_image is None:
            self.depth_image = self.current_depth
        else:
            self.depth_image = depth_image[:, :, None]


from threading import Lock
class RGBSubscriber:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback,
                                          queue_size=1)
        self.latest_image = None
        self.bridge = CvBridge()
        self.lock = Lock()

    def callback(self, msg):
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")

        with self.lock:
            self.latest_image = image

    def get_latest_image(self):
        with self.lock:
            return self.latest_image.copy() if self.latest_image is not None else None

class DepthSubscriber:
    def __init__(self, depth_max_dis):
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.callback, queue_size=1)
        self.latest_image = None
        self.bridge = CvBridge()
        self.lock = Lock()
        self.depth_max_dis = depth_max_dis

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        image = np.array(cv_image, dtype=np.float32)

        # image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
        image = image[:,100:-100] # 480x848

        image = image / 1000
        image[image == 0] = self.depth_max_dis
        image = np.clip(image, 0, self.depth_max_dis) / self.depth_max_dis

        with self.lock:
            self.latest_image = image

    def get_latest_image(self):
        with self.lock:
            return self.latest_image.copy() if self.latest_image is not None else None
