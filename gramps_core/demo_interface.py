import sys
# Enable the conda python interpreter to access ROS packages
# Even if ROS installs the packages to the system python
sys.path.append("/usr/lib/python3/dist-packages")

from time import time
from threading import Thread

from scipy.spatial.transform import Rotation as scipyR
import numpy as np

# Ros
import rospy

# Local
from .ros_arm_container import ROSArmContainer
from .state import State
from .keyboard import getch
from .addon import add_snapping_function, add_ros_subscribe_function, add_logger_function


#######################################################################
# Control loop that sends command to the arm
#######################################################################

def is_ik_jumping(state: State, command_pos):
  a = np.array(state.current_position)
  b = np.array(command_pos)
  threshold = state.ik_jumping_threshold
  if np.any(np.abs(a-b) > threshold):
    print(f"IK Jumps \t{np.array2string(np.abs(a-b), precision=4, separator=',', suppress_small=True)}")
    return True
  return False

def command_proc(state: State):
  rate = rospy.Rate(state.cmd_freq)
  while not rospy.is_shutdown():
    # Update feedback
    with state.lock:
      state.current_position, state.current_velocity, state.current_effort = state.arm.get_state()
      state.ee_pose = state.arm.fk(state.current_position)
      current_mode = state.mode

    # Generating command
    t = time()
    state.info["curr_time"] = t
    state.info["joint_pos"] = state.current_position
    state.info["robot_pose"] = state.ee_pose
    assert current_mode in state.modes
    for fn in state.pre_command_hooks["*"] + state.pre_command_hooks[current_mode]:
      fn(state)
    command_pos = state.modes[current_mode](state, t)
    for fn in state.post_command_hooks["*"] + state.post_command_hooks[current_mode]:
      fn(state)
    state.info["target_position"] = command_pos

    with state.lock:
      # Check for IK jump, apply smoother, and send out command
      if not state._mute:
        if all(pos is not None for pos in command_pos):
          if is_ik_jumping(state, command_pos):
            command_pos = state.joint_smoother.get()
          else:
            state.joint_smoother.append(command_pos)
            command_pos = state.joint_smoother.get()
        state.arm.command(command_pos)

      # Update publisher
      for publisher in state.publishers:
        publisher(state)

    rate.sleep()

###########################################################
# Key Press Handler
###########################################################

def _idle(key, state):
  state.mode = 'idle'

def _mute(key, state: State):
  with state.lock:
    state._mute = not state._mute
    state.cmd_smoother.reset()
    state.joint_smoother.reset()

def _print_state(key, state: State):
  with state.lock:
    print('current_position')
    print(state.current_position)
    print('End effector xyz rpy')
    pose_xyz_rpy = np.empty(6)
    pose_xyz_rpy[0:3] = state.ee_pose[:3]
    pose_xyz_rpy[3:6] = scipyR.from_quat(state.ee_pose[3:]).as_euler('ZYX')[::-1]
    print(str(pose_xyz_rpy))
    print('\n----\n')

def _print_help(key, state):
  print("Keypress Handlers:")
  keys = sorted(state.handlers.keys())
  for k in keys:
    print("\t%s - %s" % (k, state.handlers[k].__name__.replace("_", " ").strip()))
  print("Modes:")
  modes = sorted(state.modes.keys())
  for _mode in modes:
    print("\t%s" % _mode)

def init_default_handlers():
  handlers = {}
  handlers['z'] = _idle
  handlers['Z'] = _mute
  handlers['v'] = _print_state
  handlers['h'] = _print_help
  return handlers

# ------------------------------------------------------------------------------
# Modes Handler
# ------------------------------------------------------------------------------

def __idle(state, curr_time):
  return [None] * 7

#######################################################################
# Main thread switches running mode by accepting keyboard command
#######################################################################

def run_demo(config: dict, cmd_freq: int, callback_func=None, cli_params=None, recorded_topics=[]):
  cli_params = cli_params or {}
  arm = ROSArmContainer(config["ee_link_name"], config["arm_controller_topic"])
  state = State(config, arm, cmd_freq, cli_params)

  # Basic demo functions
  state.modes['idle'] = __idle
  state.handlers = init_default_handlers()

  # Install default handlers BEFORE custom handlers
  add_ros_subscribe_function(state, recorded_topics) # should be installed first
  add_snapping_function(state)
  add_logger_function(state)

  # Caller install custom handlers
  if callback_func is not None:
    callback_func(state)

  # command is sent out via a separate thread
  cmd_thread = Thread(target=command_proc, name='Command Thread', args=(state,))
  cmd_thread.start()

  # this script is preserved for switching modes
  np.set_printoptions(suppress=True)
  print("Press keys to invoke modes.\n")

  res = getch()
  while res != 'q' and not rospy.is_shutdown():
    print('')
    if res in state.handlers:
      try:
        state.handlers[res](res, state)
      except Exception as e:
        print(e)
    rospy.sleep(0.01)
    res = getch()

  print('Quitting...')
  rospy.signal_shutdown('User quit')
  cmd_thread.join()

  # Cleaning up
  for func in state.onclose:
    func(state)
