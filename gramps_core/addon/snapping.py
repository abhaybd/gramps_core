#######################################################################
# Snaping
# ------------------------------------------------------------------------------
# Snap the robot to a fixed set of joint position (ignoring collision!)
# - 'm' to rapidly move (3 seconds)
# - 'M' to slowly move (7 seconds)
#######################################################################

import numpy as np

from gramps_core.state import State

SLOW_MOVING_KEY = "M"
FAST_MOVING_KEY = "m"
MOVING_MODE = "moving"

def add_snapping_function(state: State):
  state.handlers[FAST_MOVING_KEY] = state.handlers[SLOW_MOVING_KEY] = _move
  state.modes[MOVING_MODE] = __move

def do_snapping(state, moving_positions, total_time, return_mode=None):
  # Utility function that makes the robot go through a set of fixed keypoints
  with state.lock:
    state.trajectory = moving_positions
    state.trajectory_start = None
    state.mode = MOVING_MODE
    state.return_mode = return_mode
    state.trajectory_duration = total_time

def _move(key, state: State):
  reset_pos = np.array(state.config["reset_pos"])
  print('Move to a predefined position ' + np.array2string(reset_pos, precision=3))
  do_snapping(state, [state.current_position, reset_pos], 7.0 if key == SLOW_MOVING_KEY else 3.0)

def __move(state: State, cur_time):
  if state.trajectory_start is None:
    print('Start moving')
    state.trajectory_start = cur_time

  elapse_time = cur_time - state.trajectory_start
  if elapse_time >= state.trajectory_duration:
    # allows other modes programmatic access to moving mode
    if state.return_mode and elapse_time >= 1.1 * state.trajectory.duration:
      print('Finish moving / snapping and return to ' + state.return_mode)
      state.mode = state.return_mode
    return state.moving_positions[-1]
  idx = np.interp(elapse_time, np.linspace(0, state.trajectory_duration, len(state.trajectory)), np.arange(len(state.trajectory)))
  pos_a = state.trajectory[int(idx)]
  pos_b = state.trajectory[int(idx) + 1]
  pos = pos_a + (pos_b - pos_a) * (idx - int(idx))
  return pos
