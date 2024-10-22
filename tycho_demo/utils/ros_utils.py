from typing import Any, Callable, Dict
import numpy as np
import importlib
from scipy.spatial.transform import Rotation as scipyR

import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped, PointStamped, QuaternionStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

TF_LISTENER = tf.TransformListener()
CV_BRIDGE = CvBridge()
ROS2NP_TRFS: Dict[type, Callable[[Any], np.ndarray]] = {}

def trf_to_pose(trf):
    p = Pose()
    q, t = p.orientation, p.position
    q.x, q.y, q.z, q.w = scipyR.from_matrix(trf[:3,:3]).as_quat()
    t.x, t.y, t.z = trf[:3,-1]
    return p

def pose_to_trf(pose):
    q, t = pose.orientation, pose.position
    trf = np.eye(4)
    quat = np.array([q.x, q.y, q.z, q.w])
    trf[:3,:3] = scipyR.from_quat(quat).as_matrix()
    trf[:3,:-1] = np.array([t.x, t.y, t.z])
    return trf

def get_frame_trf(src, dst, timeout=None):
    try:
        if timeout:
            TF_LISTENER.waitForTransform(dst, src, rospy.Time(), rospy.Duration(timeout))
        trans, rot = TF_LISTENER.lookupTransform(dst, src, rospy.Time())
    except Exception as e:
        raise e
    mat = scipyR.from_quat(rot).as_matrix()
    trf = np.eye(4)
    trf[:3,:3] = mat
    trf[:3,-1] = trans
    return trf

def apply_frame_trf(src, dst, x, timeout=None):
    if isinstance(x, Pose):
        p = Pose()
        p.position = apply_frame_trf(src, dst, x.position)
        p.orientation = apply_frame_trf(src, dst, x.orientation)
        return p
    elif isinstance(x, Point):
        trf = get_frame_trf(src, dst, timeout=timeout)
        p = Point()
        p.x, p.y, p.z = (trf @ np.array([x.x, x.y, x.z, 1.0]))[:3]
        return p
    elif isinstance(x, Quaternion):
        trf = get_frame_trf(src, dst, timeout=timeout)
        q = np.array([x.x, x.y, x.z, x.w])
        trf_q = (scipyR.from_matrix(trf[:3,:3]) * scipyR.from_quat(q)).as_quat()
        quat = Quaternion()
        quat.x, quat.y, quat.z, quat.w = trf_q
        return quat
    else:
        raise ValueError(f"Unrecognized type {type(x)}")

def numpify(msg):
    """Convert a ROS message to a numpy array. Raises a KeyError if no suitable conversion is found."""
    try:
        return ROS2NP_TRFS[type(msg)](msg)
    except KeyError:
        raise ValueError(f"Unrecognized message type {type(msg)}")

def add_ros2np_trf(msg_type: type, trf: Callable[[Any], np.ndarray]):
    """Add a custom conversion between a ROS message type and numpy array."""
    ROS2NP_TRFS[msg_type] = trf

add_ros2np_trf(PointStamped, lambda msg: numpify(msg.point))
add_ros2np_trf(QuaternionStamped, lambda msg: numpify(msg.quaternion))
add_ros2np_trf(PoseStamped, lambda msg: numpify(msg.pose))
add_ros2np_trf(Point, lambda msg: np.array([msg.x, msg.y, msg.z]))
add_ros2np_trf(Quaternion, lambda msg: np.array([msg.x, msg.y, msg.z, msg.w]))
add_ros2np_trf(Pose, lambda msg: np.concatenate([numpify(msg.position), numpify(msg.orientation)], axis=0))
add_ros2np_trf(Image, lambda msg: CV_BRIDGE.imgmsg_to_cv2(msg, desired_encoding="bgr8"))


class GenericMessageSubscriber(object):
    """
    A subscriber that doesn't need to know the message type ahead of time.
    As messages are received, they are deserialized to the correct message type, if the type is importable.
    See: https://answers.ros.org/question/36855/is-there-a-way-to-subscribe-to-a-topic-without-setting-the-type/
    """
    def __init__(self, topic_name: str, callback: Callable[[Any], None], **kwargs):
        self._binary_sub = rospy.Subscriber(
            topic_name, rospy.AnyMsg, self.generic_message_callback, **kwargs)
        self._callback = callback

    def generic_message_callback(self, data: rospy.AnyMsg):
        connection_header =  data._connection_header['type'].split('/')
        ros_pkg = connection_header[0] + '.msg'
        msg_type = connection_header[1]
        msg_class = getattr(importlib.import_module(ros_pkg), msg_type)
        msg = msg_class().deserialize(data._buff)
        self._callback(msg)

    def unregister(self):
        self._binary_sub.unregister()
