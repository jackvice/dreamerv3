# Leo Rover

ros2 launch leo_gz_bringup leo_gz.launch.py sim_world:=marsyard2020.sdf robot_ns:=leo1

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/leo1/cmd_vel

# syn dreamerv3 fork with main dreamv3 repo
git fetch upstream
git merge upstream/main
