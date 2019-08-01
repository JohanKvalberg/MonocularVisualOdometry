#!/usr/bin/env python
import h5py
import numpy as np
import rospy

from p2_drone.msg import ImuData, StateEstimate, Reference
from geometry_msgs.msg import Vector3


def print_mat():
    f = h5py.File('/home/johan/project_thesis/matlab/sensor_data.mat', 'r')

    # pos
    pos_x = f.get('pos_x')
    pos_y = f.get('pos_y')
    pos_z = f.get('pos_z')
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    pos_z = np.array(pos_z)
    # depth
    depth = f.get('depth')
    depth = np.array(depth)

    # heading
    heading = f.get('heading')
    heading = np.array(heading)

    # acceleration
    acc_x = f.get('acc_x')
    acc_y = f.get('acc_y')
    acc_z = f.get('acc_z')
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)

    # angular velocity
    gyro_x = f.get('gyro_x')
    gyro_y = f.get('gyro_y')
    gyro_z = f.get('gyro_z')
    gyro_x = np.array(gyro_x)
    gyro_y = np.array(gyro_y)
    gyro_z = np.array(gyro_z)

    measurements = np.concatenate((depth, heading, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, pos_x, pos_y, pos_z))
    return measurements


def publish_data(data):
    pub = rospy.Publisher('sensor_imu_1/data', ImuData, queue_size=10)
    pub2 = rospy.Publisher('observer/state_estimate', StateEstimate, queue_size=10)
    pub3 = rospy.Publisher('observer/pos_estimate', Reference, queue_size=10)

    rospy.init_node('data_generator')
    rate = rospy.Rate(100)

    imu = ImuData()
    imu.gyro = Vector3()
    imu.compass = Vector3()
    imu.accelerometer = Vector3()
    imu.compass.y = 0.
    imu.compass.z = 0.

    estimate = StateEstimate()
    estimate.depth_rate = 0.
    estimate.heading_rate = 0.

    reference = Reference()

    x, y = data.shape
    i = 0
    while i < y:

        estimate.depth = data[0, i]
        estimate.heading = data[1, i]
        imu.gyro.x = data[2, i]
        imu.gyro.y = data[3, i]
        imu.gyro.z = data[4, i]
        imu.accelerometer.x = data[5, i]
        imu.accelerometer.y = data[6, i]
        imu.accelerometer.z = data[7, i]
        imu.compass.z = data[1, i]
        reference.surge = data[8, i]
        reference.sway = data[9, i]
        reference.heave = data[10, i]

        i += 1

        pub.publish(imu)
        pub2.publish(estimate)
        pub3.publish(reference)
        rate.sleep()


if __name__ == '__main__':
    data = print_mat()

    try:
        publish_data(data)
    except rospy.ROSInterruptException:
        pass
