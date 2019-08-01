## Introduction
This map includes the monocular visual odometry algorihtm, as well as codes for offline tests.

## Requirements
* Python 2.7
* Numpy
* OpenCV
* ROS Kinetic¹

1) The ros nodes in this folder is also dependet on the p2_drone repository given by Blueye Robotics which has restricted access.


## Offline
An offline version of the MVO algorithm can be teset by modifing the path of video file (line 162) in `motion_estimation.py`. Keep in mind that the calibration data needs to be given and/or adjusted if testet on own video file.

To run the file
```
python motion_estimation.py
```
The results is saved in VO_results.csv


### Test on KITTI dataset
The dataset is given in:
 [KITTI odometry data set (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
 
Modify the path in KITTI_motion_estimation.py to your image sequences and ground truth trajectories, then run
```
python KITTI_motion_estimation.py
```

## With ROS
The given nodes needs to be saved in the `src` folder of in the p2_drone package. 

### MVO algorithm
To run the MVO algorithm, run the `node_motion_estimation.py` node. 

### ESKF filter
If self made measurements for simulation purposes is used, the `node_data_generator.py` publishes the data on the correct topic. The data needs to be provided in a .mat file, with the following measuremets:
```
pos_x, pos_y, pos_z, depth, heading, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
``` 

To run the accelerometer calibrator, run the `node_acc_calibrator.py` node

