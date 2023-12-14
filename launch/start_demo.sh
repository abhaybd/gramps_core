#!/bin/zsh

cd /home/prl/tycho_ws/
source $(catkin locate)/devel/setup.zsh

# 0. tmux launcher
source $(rospack find tycho_demo_ros)/launch/tmux_launcher.sh

# 1. roscore
launcher "core" "roscore"
sleep 2s

# 2. tycho description (for Rviz)
pr_ros_launcher "tycho_description" "tycho_description" "robot_chopsticks.launch"

# 3. cameras and perception

print_usage() {
  printf "\033[1;33mYou launched no camera or perception system.\033[0m\n"
  printf "To specify what perception to use: source ./start_demo.sh -o\n"
  printf "args: a(azure kinect), p(point pub from azure), o(optitrack), r(realsense)."
  printf "\n"
}

while getopts 'abor' flag; do
	case "${flag}" in
		o) source $(rospack find tycho_demo_ros)/launch/optitrack.sh ;;
		r) source $(rospack find tycho_demo_ros)/launch/realsense_camera.sh ;;
		a) source $(rospack find tycho_demo_ros)/launch/az_camera.sh ;;
		p) tmux new -d -s ball_pub "python $(rospack find tycho_demo_ros)/../tycho_perception/src/camera_ball_publisher.py" ;;
		*) print_usage ;;
	esac
done

if [ $OPTIND -eq 1 ]; then print_usage; fi

echo "Preparation done; You can view the robot in RViz. "
echo "Ready to launch demo script."
