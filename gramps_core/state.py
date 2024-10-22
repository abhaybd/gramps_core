from typing import List, Dict, Callable
from threading import Lock
from collections import defaultdict

import numpy as np

from .ros_arm_container import ROSArmContainer
from .utils import WindowSmoother, IIRFilter

class State(object):
  def __init__(self, config: dict, arm: ROSArmContainer, cmd_freq: int, cli_params: dict):
    self.mode = 'idle'
    self.config = config
    self._mute = True
    self.arm = arm
    self.cmd_freq = cmd_freq
    self.cli_params = cli_params
    self.publishers = []

    # modes and hooks
    # Invoked on the control thread to get the current command (position and velocity setpoints for each joint)
    self.modes: Dict[str, Callable[[State, float], tuple[np.ndarray, np.ndarray]]] = {}
    # Invoked on the control thread when handling input
    self.handlers: Dict[str, Callable[[str, State], None]] = {}
    # Invoked when shutting down the program
    self.onclose: List[Callable[[State], None]] = []
    # The following hooks are called from the command thread before and after querying the mode callback
    self.pre_command_hooks: Dict[str, List[Callable[[State], None]]] = defaultdict(list)
    self.post_command_hooks: Dict[str, List[Callable[[State], None]]] = defaultdict(list)
    self.info = {} # everything here must be pickleable

    self.current_position, self.current_velocity, self.current_effort = arm.get_state()
    self.ee_pose = arm.fk(self.current_position)
    self.cmd_pose = self.ee_pose

    self.cmd_smoother = WindowSmoother(arm.n_joints, 30)
    self.joint_smoother = IIRFilter(np.array([1., 0.75, 0.4, 0.2, 1., 0.2, 1.]))

    # For threading safety
    self.lock = Lock()
