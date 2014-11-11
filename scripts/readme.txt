Steps to running code:
Start roscore
Ensure drive_ifsmiles_v3.py is an executable
Make sure you have the neato_node stuff from earlier projects
Start neato node (roslaunch neato_node bringup.launch host:=IP_address_of_py
Run code: rosrun VisionProject drive_ifsmiles_v3.py

If you want to debug the vision code, run rosrun image_view image_view image:=/camera/image_raw

Be sure to first download and extract smile database found here (UCSD) https://drive.google.com/file/d/0B0UHkPLHsgyoczV6ZGx6Wnh3TjQ/edit?usp=sharing
Good reference for how that works from DataScience Class @ Olin: https://sites.google.com/site/datascience14/lectures/lecture-14

Important chunks of code:
train_smile.py (runs through database) (if you want to run through a different database you need to modify load_smiles.py)
webcam_modified_fixed.py (runs smile detection on face detected from laptop camera feed)
smileDetector_neato.py analyzes neato image feed but it didn't seem to work
