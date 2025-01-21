# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.
        self.e_psi_prev = 0
        self.e_dist_prev = 0

        # Member variables for longitudinal PID controller
        self.Kp_long = 100
        self.Kd_long = 0.1
        self.Ki_long = 0.1
        self.cum_error_long = 0
        self.prev_error_long = 0

        self.lookahead_long = 75
        self.lookahead_lat = 200
        self.horizon = 100
        self.Q = np.array([[15, 0, 0, 0],
                           [0, 0.1, 0, 0],
                           [0, 0, 3000, 0],
                           [0, 0, 0, 400]])
        # self.Qf = np.array([[0.1, 0, 0, 0],
        #                    [0, 0.1, 0, 0],
        #                    [0, 0, 0.1, 0],
        #                    [0, 0, 0, 0.1]])
        self.Qf = self.Q
        self.R = np.array([3])

    def F_PID(self, error_long, delT):
        self.cum_error_long += error_long * delT  # adding current error to cumulative error
        d_error_long = (error_long - self.prev_error_long) / delT  # computing error derivative from current and previous error
        self.prev_error_long = error_long  # updating previous error
        
        F = self.Kp_long * error_long + self.Ki_long * self.cum_error_long + self.Kd_long * d_error_long  # longitudinal PID controller
        F = clamp(F, 0, 15736)  # setting F to be within the specified limits

        return F
    
    def LQR(self, A, B, Q, Qf, R, delT):
        Ad = linalg.expm(A*delT)
        Bd = B*delT
        S = np.zeros((self.horizon + 1, 4, 4))
        S[-1] = Qf

        for i in range(self.horizon - 1, -1, -1):
            S_ = Ad.T@S[i + 1]@Ad - Ad.T@S[i + 1]@Bd @ np.linalg.inv(R + Bd.T@S[i + 1]@Bd) @ Bd.T@S[i + 1]@Ad + Q
            S[i] = S_

        K = np.zeros([1, 4])

        for i in range(10):
            K += -np.linalg.inv(R + Bd.T@S[1]@Bd) @ Bd.T @ S[1] @ Ad
        K = K/10
        return K
    


    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        lookahead_long = self.lookahead_long
        lookahead_lat = self.lookahead_lat

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        _, min_dist_idx = closestNode(X, Y, trajectory)

        if (min_dist_idx + lookahead_lat >= 8203):
            lookahead_lat = 8203 - min_dist_idx - 1
        if (min_dist_idx + lookahead_long >= 8203):
            lookahead_long = 8203 - min_dist_idx - 1

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        e_x = trajectory[min_dist_idx + lookahead_lat, 0] - X
        e_y = trajectory[min_dist_idx + lookahead_lat, 1] - Y

        e_psi = wrapToPi(psi - np.atan2(e_y, e_x))
        e_psi_dot = psidot

        e_dist = np.sqrt(e_x**2 + e_y**2) * e_psi
        e_dist_dot = ydot + xdot*e_psi

        error_lat = np.hstack((e_dist, e_dist_dot, e_psi, e_psi_dot))

        A_lat = np.matrix([[0, 1, 0, 0],
                          [0, -4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf - lr)/(m*xdot)],
                          [0, 0, 0, 1],
                          [0, -2*Ca*(lf - lr)/(Iz*xdot), 2*Ca*(lf - lr)/Iz, -2*Ca*(lf**2 + lr**2)/(Iz*xdot)]])
        B_lat = np.matrix([[0],
                          [2*Ca/m],
                          [0],
                          [2*Ca*lf/Iz]])

        K = self.LQR(A_lat, B_lat, self.Q, self.Qf, self.R, delT)
        u = np.dot(K, error_lat)
        delta = float(u)
        delta = clamp(delta, -np.pi/6, np.pi/6)

        # ---------------|Longitudinal Controller|-------------------------
        e_long = 40 - xdot
        F = self.F_PID(np.abs(e_long), delT)


        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
