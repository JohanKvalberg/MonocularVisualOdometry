#!/usr/bin/env python

import rospy
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

from errorStateKalmanFilter import ErrorStateKalmanFilter, plot, log_data, quaternion2rad
from p2_drone.msg import ImuData, StateEstimate, Reference


class Calibrator:
    def __init__(self):
        self.pos = np.zeros((3, 1))
        self.depth = None
        self.depth_rate = None
        self.heading = 0.
        self.heading_rate = None
        self.observer_time = None
        self.accelerometer = np.zeros((3, 1))
        self.gyro = np.zeros((3, 1))
        self.compass = np.zeros((3, 1))
        self.imu_time = time.time()
        self.imu_dt = 0.

    def estimates(self, state_estimate):
        self.depth = state_estimate.depth
        self.depth_rate = state_estimate.depth_rate
        self.heading = state_estimate.heading
        self.heading_rate = state_estimate.heading_rate
        self.observer_time = time.time()

    def imu_data(self, data):
        self.imu_dt = time.time() - self.imu_time
        self.imu_time = time.time()
        self.accelerometer[0, 0] = data.accelerometer.x
        self.accelerometer[1, 0] = data.accelerometer.y
        self.accelerometer[2, 0] = data.accelerometer.z
        self.gyro[0, 0] = data.gyro.x
        self.gyro[1, 0] = data.gyro.y
        self.gyro[2, 0] = data.gyro.z
        self.compass[0, 0] = data.compass.x
        self.compass[1, 0] = data.compass.y
        self.compass[2, 0] = data.compass.z
        #z = np.concatenate((np.array([[self.depth]]), np.array([[self.heading]]), self.accelerometer, self.gyro))
        #eskf.update(self.accelerometer, self.gyro, z, self.imu_dt)

    def pos_data(self, pos):
        self.pos[0, 0] = pos.surge
        self.pos[1, 0] = pos.sway
        self.pos[2, 0] = pos.heave


def acc_calibrator():

    calibrator = Calibrator()

    rospy.Subscriber('observer/state_estimate', StateEstimate, calibrator.estimates)
    rospy.Subscriber('sensor_imu_1/data', ImuData, calibrator.imu_data)
    rospy.Subscriber('observer/pos_estimate', Reference, calibrator.pos_data)
    pub = rospy.Publisher('accelerometer/acc_calibrated', ImuData, queue_size=10)
    rospy.init_node('acc_calibrator')

    # Error State Kalman Filter (ESKF)
    eskf = ErrorStateKalmanFilter()

    data = ImuData()

    # Plot lists
    xs = []
    acc_x = []
    acc_y = []
    acc_z = []
    acc_bias_x = []
    acc_bias_y = []
    acc_bias_z = []

    row = ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
           'a_x', 'a_y', 'a_z', 'd_x', 'd_y', 'd_z', 'b_x', 'b_y', 'b_z', 'Time', 'a_x_raw', 'a_y_raw', 'a_z_raw']
    f = open('eskf_data.csv', 'w+')
    writer = csv.writer(f)
    writer.writerow(row)

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        calibrator.pos[2, 0] = calibrator.depth
        z = np.concatenate((calibrator.pos, np.array([[calibrator.heading]]), calibrator.gyro,
                            calibrator.accelerometer))
        eskf.update(calibrator.accelerometer, calibrator.gyro, z, calibrator.imu_dt)

        data.accelerometer.x = eskf.x_hat[13, 0]
        data.accelerometer.y = eskf.x_hat[14, 0]
        data.accelerometer.z = eskf.x_hat[15, 0]
        data.gyro.x = calibrator.gyro[0, 0]
        data.gyro.y = calibrator.gyro[1, 0]
        data.gyro.z = calibrator.gyro[2, 0]
        data.compass.x = calibrator.compass[0, 0]
        data.compass.y = calibrator.compass[1, 0]
        data.compass.z = calibrator.compass[2, 0]

        pub.publish(data)

        row = eskf.x_hat[:, 0].tolist()
        orientation = quaternion2rad(eskf.x_hat[3:7])
        del row[3]
        row[3:6] = orientation[:, 0]*180/np.pi
        row.append(eskf.time)
        row.append(calibrator.accelerometer[0, 0])
        row.append(calibrator.accelerometer[1, 0])
        row.append(calibrator.accelerometer[2, 0])
        writer.writerow(row)
        
        rate.sleep()

    f.close()


if __name__ == '__main__':
    try:
        acc_calibrator()
    except rospy.ROSInterruptException:
        pass
