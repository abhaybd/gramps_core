from threading import Lock
from typing import Tuple

import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from gramps_kd.srv import ForwardKinematics, InverseKinematics


class ROSArmContainer(object):
    def __init__(self, ee_link_name: str, controller_ns: str):
        self._ee_link_name = ee_link_name

        self._state_lock = Lock()
        joint_states: JointState = rospy.wait_for_message(f"{controller_ns}/joint_states", JointState, timeout=5)
        self._joint_pos: np.ndarray = np.array(joint_states.position)
        self._joint_vel: np.ndarray = np.array(joint_states.velocity)
        self._joint_eff: np.ndarray = np.array(joint_states.effort)

        self._state_pub = rospy.Subscriber(f"{controller_ns}/joint_states", JointState, self._joint_state_callback, queue_size=10)
        self._command_pub = rospy.Publisher(f"{controller_ns}/command", Float64MultiArray, queue_size=10)

        rospy.wait_for_service("forward_kinematics", timeout=5)
        rospy.wait_for_service("inverse_kinematics", timeout=5)
        self._fk = rospy.ServiceProxy("forward_kinematics", ForwardKinematics, persistent=True)
        self._ik = rospy.ServiceProxy("inverse_kinematics", InverseKinematics, persistent=True)

    @property
    def n_joints(self):
        return len(self.joint_pos)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns current joint positions, velocities, and efforts"""
        with self._state_lock:
            return self._joint_pos.copy(), self._joint_vel.copy(), self._joint_eff.copy()
    
    def _joint_state_callback(self, msg: JointState):
        with self._state_lock:
            self.joint_pos = np.array(msg.position)
            self.joint_vel = np.array(msg.velocity)
            self.joint_eff = np.array(msg.effort)

    def command(self, joint_pos: list):
        assert len(joint_pos) == self.n_joints, f"Joint positions must be a {self.n_joints}-element array"
        joint_pos = [x if x is not None else np.nan for x in joint_pos]
        joint_pos_msg = Float64MultiArray(data=joint_pos)
        self._command_pub.publish(joint_pos_msg)

    def fk(self, joint_pos: list) -> np.ndarray:
        """Returns end-effector pose as [xyz, quat, gripper]"""
        assert len(joint_pos) == self.n_joints, f"Joint positions must be a {self.n_joints}-element array"
        fk_res = self._fk(Float64MultiArray(data=joint_pos), self._ee_link_name)
        return np.array([
            fk_res.pose.position.x,
            fk_res.pose.position.y,
            fk_res.pose.position.z,
            fk_res.pose.orientation.x,
            fk_res.pose.orientation.y,
            fk_res.pose.orientation.z,
            fk_res.pose.orientation.w,
            joint_pos[-1]
        ])

    def ik(self, pose: list, joint_pos: list) -> np.ndarray:
        """pose is [xyz, quat, gripper]"""
        assert len(pose) == 8, "Pose must be a 8-element array"
        assert len(joint_pos) == self.n_joints, f"Joint positions must be a {self.n_joints}-element array"
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]
        pose_msg.position.z = pose[2]
        pose_msg.orientation.x = pose[3]
        pose_msg.orientation.y = pose[4]
        pose_msg.orientation.z = pose[5]
        pose_msg.orientation.w = pose[6]
        ik_res = self._ik(self._ee_link_name, pose, Float64MultiArray(data=joint_pos))
        return np.concatenate([np.array(ik_res.q[:self.n_joints-1]), pose[-1:]])
