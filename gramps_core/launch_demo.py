import sys; sys.path.append("/usr/lib/python3/dist-packages")
import rospy

from gramps_core.demo_interface import run_demo
from gramps_core.addon import add_tuning_function, add_replay_function

def handler_installer(state):
  add_replay_function(state)
  add_tuning_function(state)

if __name__ == '__main__':
  rospy.init_node("tycho_demo")
  run_demo(callback_func=handler_installer, cmd_freq=20) # TODO fix
