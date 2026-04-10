# PX4 SLAM with GTSAM

~TODO: add instructions~  
TODO: add camera  
TODO: add gps  

got imu preintegration working with GTSAM:  
![video](https://github.com/user-attachments/assets/12989f19-69b0-449a-b3e2-be73838d3818)

## dev setup
we are working with px4 v1.16

the default docs will take you to the docs for the main branch, but that is not
what we want to do.

*https://docs.px4.io/v1.16/en/dev_setup/dev_env_linux_ubuntu*  
vs.  
*https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu*  

we always want to look at pages with v1.16 in the URL

### install ros2 jazzy  
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

### setup ros2 workspace  
make a workspace:  
```bash  
mkdir -p ~/repos/ws/src/  
cd ~/repos/ws/src/  
git clone -b v1.16.1 https://github.com/PX4/px4_msgs.git  
cd ~/repos/ws/  
colcon build # takes ~3 min  
```

### setup this project
```bash
cd ~/repos/ws/src/  
git clone https://github.com/gjcliff/px4_slam.git
cd px4_slam
pip install -r requirements.txt
cd ~/repos/ws/
colcon build
```

### setup px4  
follow instructions here:  
https://docs.px4.io/v1.16/en/dev_setup/dev_env_linux_ubuntu

in a nutshell:  
```bash  
cd ~/repos/
git clone -b release/v1.16 https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
sudo reboot
```

after reboot, go back and build
```bash
cd ~/repos/PX4-Autopilot
make px4_sitl
```

install uxrce-dds-bridge
```bash
sudo snap install micro-xrce-dds-agent --edge # specifically this version
```

### running everything
you'll need three terminals

terminal 1:
```bash
# run this in any directory
micro-xrce-dds-agent udp4 -p 8888                                                                                                                                                [18:38:07]
```

terminal 2:
```bash
cd ~/repos/PX4-Autopilot
make px4_sitl gz_x500
```

terminal 3:
```bash
cd ~/repos/ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run px4_slam gtsam
```
