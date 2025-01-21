# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

        # Add additional member variables according to your need here.
        self.e_psi_prev = 0
        self.e_dist_prev = 0

        # Member variables for longitudinal PID controller
        self.Kp_long = 110
        self.Kd_long = 0.1
        self.Ki_long = 0.001
        self.cum_error_long = 0
        self.prev_error_long = 0

        self.lookahead = 30
    
    def F_PID(self, error_long, delT):
        self.cum_error_long += error_long * delT  # adding current error to cumulative error
        d_error_long = (error_long - self.prev_error_long) / delT  # computing error derivative from current and previous error
        self.prev_error_long = error_long  # updating previous error
        
        F = self.Kp_long * error_long + self.Ki_long * self.cum_error_long + self.Kd_long * d_error_long  # longitudinal PID controller
        F = clamp(F, 0, 15736)  # setting F to be within the specified limits

        return F

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        lookahead = self.lookahead

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        _, min_dist_idx = closestNode(X, Y, trajectory)

        if (min_dist_idx + lookahead >= 8203):
            lookahead = 8203 - min_dist_idx - 1

        e_x = trajectory[min_dist_idx + lookahead, 0] - X
        e_y = trajectory[min_dist_idx + lookahead, 1] - Y

        e_psi = wrapToPi(psi - np.atan2(e_y, e_x))
        e_psi_dot = psidot

        e_dist = np.sqrt(e_x**2 + e_y**2) * e_psi
        e_dist_dot = ydot + xdot*e_psi

        error = np.hstack((e_dist, e_dist_dot, e_psi, e_psi_dot))

        # ---------------|Lateral Controller|-------------------------
        A_lat = np.matrix([[0, 1, 0, 0],
                          [0, -4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf - lr)/(m*xdot)],
                          [0, 0, 0, 1],
                          [0, -2*Ca*(lf - lr)/(Iz*xdot), 2*Ca*(lf - lr)/Iz, -2*Ca*(lf**2 + lr**2)/(Iz*xdot)]])
        B_lat = np.matrix([[0],
                          [2*Ca/m],
                          [0],
                          [2*Ca*lf/Iz]])
        poles = np.array([-31, -10, -5.1, -0.001])
        result = signal.place_poles(A_lat, B_lat, poles)
        K = result.gain_matrix

        delta = float(np.matmul(-K, error))

        # ---------------|Longitudinal Controller|-------------------------
        # e_long = np.sqrt(e_x**2 + e_y**2)
        e_long = 30 - xdot
        F = self.F_PID(np.abs(e_long), delT)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
