VisionProject
=============

Code Repo for Project 2 in Computational Robotics

This code detects people's faces from a webcam and sends a command to drive the neato robot forwards or backwards

Code is dependent (for communicating with Neato) on earlier ROS packages from the class, such as neato_node.  

The main thing to run is drive_ifsmiles.py -- since this requires ROS to run, you must have ROS Hydro installed.
To run the file: roscore
roslaunch neato_node bringup.launch host:=PI_IP_ADDRESS
rosrun VisionProject drive_ifsmiles_v3.py

Depending on your setup you may need to tweak the number of nearest neighbors in the face detection algorithm0
which is located in drive_ifsmiles_v3 under faces variable, or you may need to edit the threshold values for smiling/not smiling
