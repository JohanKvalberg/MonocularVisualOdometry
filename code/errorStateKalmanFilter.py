import numpy as np
import rospy
import matplotlib.pyplot as plt

# State vector x_hat = [p, q, v, w, a, d, b]'           Here q is given in quaternions
# State error vector delta_x = [p, q, v, w, a]'         Here q is in the form of a rotation vector

ANGULAR_RATE_EPSILON = 1e-3


class ErrorStateKalmanFilter:
    def __init__(self):
        self.A = None
        self.B = None
        self.H = np.zeros((10, 15))
        self.K = None
        self.P_hat = np.identity(15)
        self.P_bar = np.identity(15)
        self.F = None
        self.R = np.zeros((10, 10))                 # Measurement noise covariance
        self.Q = np.identity(15)*1.0e-2             # Process noise covariance
        self.std = 1/60
        self.g = np.array([[0.], [0.], [9.81]])
        self.x_hat = np.zeros((22, 1))
        self.x_hat[3, 0] = 1
        self.delta_x = np.zeros((15, 1))
        self.Rot = np.identity(3)
        self.time = 0.
        self.init_bias = False

    def update(self, acc, omega, z, dt):
        self.time += dt
        z[3:7, 0] = z[3:7, 0]*np.pi/180

        self.Rot = rotation_matrix(z[1])
        z[7:10] = z[7:10] - self.Rot.dot(self.g) - self.x_hat[19:22]

        # Variance
        pos_var = 1.0e-08
        depth_var = 1.0e-6
        heading_var = 1.0e-6
        acc_var = 9.5e-04
        acc_bias_var = 6.7502e-06
        gyro_var = 9.5e-10
        gyro_drift_var = 6.7502e-8

        self.set_covariance_matrix(acc_var, acc_bias_var, gyro_var, gyro_drift_var, dt)
        self.R[0:2, 0:2] = np.eye(2)*pos_var
        self.R[2, 2] = depth_var
        self.R[3, 3] = heading_var
        self.R[4:7, 4:7] = gyro_var*np.identity(3)
        self.R[7:10, 7:10] = acc_var*np.identity(3)

        # ---- Nominal state prediction ----
        # Estimated states, x_hat
        # 0 position = x_hat[0:3]
        # 1 orientation = x_hat[3:7] (given in quaternions)
        # 2 lin velocity = x_hat[7:10]
        # 3 ang velocity = x_hat[10:13]
        # 4 acceleration = x_hat[13:16]
        # 5 gyro drift = x_hat[16:19]
        # 6 acceleration bias = x_hat[19:22]

        self.x_hat[0:3] = self.x_hat[0:3] + self.Rot.dot(self.x_hat[7:10])*dt + 0.5*self.Rot.dot(self.x_hat[13:16])*(dt**2)
        self.x_hat[3:7] = quaternion_multiply(quaternion(omega, dt), self.x_hat[3:7])
        self.x_hat[7:10] = self.x_hat[7:10] + self.x_hat[13:16]*dt
        self.x_hat[10:13] = omega #- self.x_hat[16:19]
        self.x_hat[13:16] = np.transpose(self.Rot).dot(self.g) - acc + self.x_hat[19:22]
        self.x_hat[16:19] = self.x_hat[16:19]
        self.x_hat[19:22] = self.x_hat[19:22]


        # ---- Error state prediction ----
        # error states, delta_x
        # 0 position error = delta_x[0:3]
        # 1 orientation error = delta_x[3:6]
        # 2 lin velocity error = delta_x[6:9]
        # 3 ang velocity error = delta_x[9:12]
        # 4 acceleration error = delta_x[12:15]

        # Is the error state is zero (which it is after the reset), the states will not change.
        # However, here are the equations:
        self.delta_x[0:3] = self.delta_x[0:3] + self.Rot.dot(self.delta_x[6:9])*dt - self.Rot.dot(cross(self.x_hat[7:10], self.delta_x[3:6]))*dt \
                          - 0.5*self.Rot.dot(cross(self.x_hat[13:16], self.delta_x[3:6]))*(dt**2) + 0.5*self.Rot.dot(self.delta_x[12:15])*(dt**2)

        self.delta_x[6:9] = self.delta_x[6:9] + cross(np.transpose(self.Rot).dot(self.g), self.delta_x[3:6])*dt + self.delta_x[12:15]*dt

		#self.delta_x[3:6] = TH.dot(self.delta_x[3:6]) + PS.dot(self.delta_x[6:9])

        #  Update the error state derivative matrix.
        rate = np.linalg.norm(self.x_hat[10:13])
        angle = dt*rate
        s = skew(self.x_hat[7:10])
        TH = np.transpose(angle_axis(angle, normalize(self.x_hat[10:13])))
        if rate < ANGULAR_RATE_EPSILON:
            PS = -dt * np.identity(3) + (dt**2)*s/2 - (dt**3)*(s**2)/6
        else:
            PS = -dt * np.identity(3) + ((1 - np.cos(angle))/rate**2)*s - ((angle - np.sin(angle))/rate**3)*(s**2)

        dp_dp = np.identity(3)
        dp_dq = -dt*self.Rot.dot(skew(self.x_hat[7:10])) + 0.5*(dt**2)*self.Rot.dot(skew(acc - self.x_hat[19:22]))
        dp_dv = dt*self.Rot
        dp_da = 0.5*(dt**2)*self.Rot
        dq_dq = TH
        dq_dw = PS
        dv_dq = dt*skew(np.transpose(self.Rot).dot(self.g))
        dv_dv = np.identity(3)
        dv_da = dt*np.identity(3)
        dw_dw = np.identity(3)
        da_da = np.identity(3)

        self.F = np.zeros((15, 15))
        self.F[0:3, 0:3] = dp_dp
        self.F[0:3, 3:6] = dp_dq
        self.F[0:3, 6:9] = dp_dv
        self.F[0:3, 12:15] = dp_da
        self.F[3:6, 3:6] = dq_dq
        self.F[3:6, 9:12] = dq_dw
        self.F[6:9, 3:6] = dv_dq
        self.F[6:9, 6:9] = dv_dv
        self.F[6:9, 12:15] = dv_da
        self.F[9:12, 9:12] = dw_dw
        self.F[12:15, 12:15] = da_da

        # ------ Update the error state 
        self.delta_x[3:6] = TH.dot(self.delta_x[3:6]) + PS.dot(self.delta_x[6:9])
        # --------------------------------------------

        self.H[0:3, 0:3] = np.eye(3)*1.         # depth p_z
        self.H[3, 5] = 1.                       # heading q_phi
        self.H[4:7, 9:12] = np.identity(3)      # gyro
        self.H[7:10, 12:15] = np.identity(3)    # accelerometer

        # -----------------------------------------------
        # -------------- Filter Equations ---------------

        # measurement error
        attitude = quaternion2rad(self.x_hat[3:7])
        z_hat = np.array([[self.x_hat[0, 0]], [self.x_hat[1, 0]], [self.x_hat[2, 0]], [attitude[2, 0]], [self.x_hat[10, 0]], [self.x_hat[11, 0]],
                          [self.x_hat[12, 0]], [self.x_hat[13, 0]], [self.x_hat[14, 0]], [self.x_hat[15, 0]]])
        delta_z = z - z_hat

        S = (self.H.dot(self.P_bar)).dot(np.transpose(self.H)) + self.R        
        self.K = (self.P_bar.dot(self.H.transpose())).dot(np.linalg.inv(S))

        # update estimate and covariance matrix
        self.delta_x = self.delta_x + self.K.dot(delta_z - self.H.dot(self.delta_x))
        self.P_hat = (np.identity(15) - self.K.dot(self.H)).dot(self.P_bar)
        # rospy.loginfo(self.P_hat)

        self.P_bar = (self.F.dot(self.P_hat)).dot(np.transpose(self.F)) + self.Q

        # -------------------------------------------------

        # ---- Nominal state correction and error reset ----
        self.x_hat[0:3] += self.delta_x[0:3]
        self.x_hat[3:7] = quaternion_multiply(self.x_hat[3:7], rad2quaternion(self.delta_x[3:6]))
        self.x_hat[7:10] += self.delta_x[6:9]
        self.x_hat[10:13] -= self.delta_x[9:12]
        self.x_hat[13:16] += self.delta_x[12:15]
        self.x_hat[16:19] += self.delta_x[9:12]
        self.x_hat[19:22] += self.delta_x[12:15]
        #rospy.loginfo(self.x_hat)

        self.delta_x[:] = 0

    def set_covariance_matrix(self, acc_var, acc_bias_var, gyro_var, gyro_drift_var, dt):
        # self.Q = np.zeros((15, 15))
        # I = np.identity(3)
        # rate = np.linalg.norm(self.x_hat[10:13])
        # angle = dt*rate
        # s = skew(self.x_hat[10:13])
        #
        # q_pp = 0.5 * (dt**2) * acc_var * I
        # q_vv = dt * acc_var * I
        # q_aa = dt * acc_bias_var * I
        # q_ww = dt * gyro_drift_var * I
        #
        # if rate < ANGULAR_RATE_EPSILON:
        #     q_qq = dt*gyro_var*I + gyro_drift_var*(((dt**3)/3.0)*I + ((dt**5)/60.0)*(s**2))
        #     q_qw = -gyro_drift_var*((dt**2)/2 * I - (dt**3)/6 * s + (dt**4)/24.0 *(s**2))
        # else:
        #     q_qq = dt*gyro_var*I + gyro_drift_var*(((dt**3)/3.0)*I + (angle**3/3.0 + 2.0*(np.sin(angle) - angle))/(rate**5)*(s**2))
        #     q_qw = -gyro_drift_var*((dt**2)/2 * I - (angle - np.sin(angle))/(rate**3)*s + (angle**2/2.0 + np.cos(angle - 1))/rate**4 * (s**2))
        #
        # self.Q[0:3, 0:3] = q_pp
        # self.Q[3:6, 3:6] = q_qq
        # # self.Q[5, 5] = 0
        # self.Q[6:9, 6:9] = q_vv
        # self.Q[9:12, 9:12] = q_ww
        # self.Q[12:15, 12:15] = q_aa
        # self.Q[3:6, 9:12] = q_qw
        # self.Q[9:12, 3:6] = q_qw
        
        self.Q[0:3, 0:3] = 4.75e-10 * np.identity(3)        # q_pp
        self.Q[3:6, 3:6] = 9.50e-08 * np.identity(3)        # q_qq
        self.Q[6:9, 6:9] = 9.50e-08 * np.identity(3)        # q_vv

        self.Q[9:12, 9:12] = 6.7502e-10 * np.identity(3)    # q_ww
        self.Q[12:15, 12:15] = 6.7502e-07 * np.identity(3)  # q_aa

    def setH(self):
        q_x, q_y, q_z, q_w = self.x_hat[3:7]
        Q_deltaT = 0.5*np.array([[-q_x, -q_y, -q_z],
                                 [ q_w, -q_z,  q_y],
                                 [ q_z,  q_w, -q_x],
                                 [-q_y,  q_x,  q_w]])

        X_delta_x = np.eye(19)
        X_delta_x[6:10] = Q_deltaT

# -------------- Additional functions ----------------------

def cross(a, b):
    x1, y1 = a.shape
    x2, y2 = b.shape
    if y1 > x1 and y2 > x2:
        return np.transpose(np.cross(a, b))
    elif y1 < x1 and y2 > x2:
        return np.transpose(np.cross(np.transpose(a), b))
    elif y1 > x1 and y2 < x2:
        return np.transpose(np.cross(a, np.transpose(b)))
    elif y1 < x1 and y2 < x2:
        return np.transpose(np.cross(np.transpose(a), np.transpose(b)))


def skew(omega):
    # computes the skew matrix, a x b = S(a)*b
    return np.array([[0, -omega[2, 0], omega[1, 0]],
                     [omega[2, 0], 0, -omega[0, 0]],
                     [-omega[1, 0], omega[0, 0], 0]])


def noise():
    return np.array([[np.random.normal(0, 1/60), 0, 0],
                     [0, np.random.normal(0, 1/60), 0],
                     [0, 0, np.random.normal(0, 1/60)]])


def quaternion(omega, t):
    cy = np.cos(omega[0, 0]*t/2)
    sy = np.sin(omega[0, 0]*t/2)
    cp = np.cos(omega[1, 0]*t/2)
    sp = np.sin(omega[1, 0]*t/2)
    cr = np.cos(omega[2, 0]*t/2)
    sr = np.sin(omega[2, 0]*t/2)

    return np.array([[cy*cp*cr + sy*sp*sr],
                     [sy*cp*cr - cy*sp*sr],
                     [cy*sp*cr + sy*cp*sr],
                     [cy*cp*sr - sy*sp*cr]])


def rad2quaternion(attitude):
    return quaternion(attitude, 1)


def quaternion2rad(attitude):
    w = attitude[0, 0]
    x = attitude[1, 0]
    y = attitude[2, 0]
    z = attitude[3, 0]

    # roll (x-axis rotation)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2*(w*y - z*x)
    if np.absolute(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([[roll], [pitch], [yaw]])


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([w2*w1 - x2*x1 - y2*y1 - z2*z1,
                     w2*x1 + x2*w1 - y2*z1 + z2*y1,
                     w2*y1 + x2*z1 + y2*w1 - z2*x1,
                     w2*z1 - x2*y1 + y2*x1 + z2*w1])


def angle_axis(angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x = axis[0, 0]
    y = axis[1, 0]
    z = axis[2, 0]
    return np.array([[t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
                     [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
                     [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rotation_matrix(q):
    # Rotation matrix from {b} to {n}
    if q.size > 1:
        return np.array([[np.cos(q[2, 0])*np.cos(q[1, 0]), -np.sin(q[2])*np.cos(q[0, 0]) + np.cos(q[2, 0])*np.sin(q[1, 0])*np.sin(q[0, 0]),  np.sin(q[2, 0])*np.sin(q[0, 0]) + np.cos(q[2, 0])*np.cos(q[0, 0])*np.sin(q[1, 0])],
                         [np.sin(q[2, 0])*np.cos(q[1, 0]),  np.cos(q[2])*np.cos(q[0, 0]) + np.sin(q[0, 0])*np.sin(q[1, 0])*np.sin(q[2, 0]), -np.cos(q[2, 0])*np.sin(q[0, 0]) + np.sin(q[1, 0])*np.sin(q[2, 0])*np.sin(q[0, 0])],
                         [-np.sin(q[1, 0]),                           np.cos(q[1, 0])*np.sin(q[0, 0]),                                                 np.cos(q[1, 0])*np.cos(q[0, 0])]])
    # using the assumption that roll and pitch is small
    c = np.cos(q)
    s = np.sin(q)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def plot(xs, acc_x, acc_y, acc_z, acc_bias_x, acc_bias_y, acc_bias_z):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    a_x, = ax.plot(xs, acc_x, label='acceleration x-direction')
    a_y, = ax.plot(xs, acc_y, label='acceleration y-direction')
    a_z, = ax.plot(xs, acc_z, label='acceleration z-direction')

    a_b_x, = ax.plot(xs, acc_bias_x, label='acceleration bias x')
    a_b_y, = ax.plot(xs, acc_bias_y, label='acceleration bias y')
    a_b_z, = ax.plot(xs, acc_bias_z, label='acceleration bias z')

    plt.title('Acceleration in x-direction')
    plt.ylabel('Acceleration [m/s]')
    plt.legend(handles=[a_x, a_y, a_z, a_b_x, a_b_y, a_b_z])
    plt.show(block=True)

