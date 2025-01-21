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
        
        # Initiating variables to store cumulative and previous errors
        self.cum_error_lat = 0
        self.cum_error_long = 0
        self.prev_error_lat = 0
        self.prev_error_long = 0
        
        # PID gains for lateral controller
        self.Kp_lat = 0.5
        self.Kd_lat = 0.015
        self.Ki_lat = 0.25

        # PID gains for longitudinal controller
        self.Kp_long = 100
        self.Kd_long = 0.01
        self.Ki_long = 0.1
        
        # Lookahead constants that dictate how far into the future the controller looks
        self.lookahead_lat = 100  # lookahead for lateral controller
        self.lookahead_long = 50  # lookahead for longitudinal controller

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        
        # cumulative and previous errors
        cum_error_lat = self.cum_error_lat
        cum_error_long = self.cum_error_long
        prev_error_lat = self.prev_error_lat
        prev_error_long = self.prev_error_long
        
        # gain constants for lateral controller
        Kp_lat = self.Kp_lat
        Kd_lat = self.Kd_lat
        Ki_lat = self.Ki_lat
        
        # gain constants for longitudinal controller
        Kp_long = self.Kp_long
        Kd_long = self.Kd_long
        Ki_long = self.Ki_long
        
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        # getting the closest waypoint from the current position
        min_dist, min_idx = closestNode(X, Y, trajectory)

        # ---------------|Lateral Controller|-------------------------

        error_lat = 0  # initializing heading error
        
        # Looking ahead and averaging out the error
        for i in range(self.lookahead_lat):
            # checks to ensure that the lookahead doesn't cause out-of-bounds access
            if(min_idx + self.lookahead_lat < len(trajectory)):
                error_Y = trajectory[min_idx + i][1] - Y
                error_X = trajectory[min_idx + i][0] - X
            else:
                error_Y = trajectory[len(trajectory) - 1][1] - Y
                error_X = trajectory[len(trajectory) - 1][0] - X
        
            error_lat += wrapToPi(np.atan2(error_Y, error_X) - psi)

        error_lat = error_lat/self.lookahead_lat  # computing the mean lateral error in terms of heading
                
        cum_error_lat += error_lat * delT  # adding current error to cumulative error
        d_error_lat = (error_lat - prev_error_lat) / delT  # computing error derivative from current and previous error
        prev_error_lat = error_lat  # updating previous error
        
        delta = Kp_lat * error_lat + Ki_lat * cum_error_lat + Kd_lat * d_error_lat  # lateral PID controller
        
        delta = clamp(delta, -np.pi/6, np.pi/6)  # setting delta to be within the specified limits


        # ---------------|Longitudinal Controller|-------------------------

        error_long = 0  # initializing longitudinal error

        # checks to ensure that the lookahead doesn't cause out-of-bounds access        
        if(min_idx + self.lookahead_long < len(trajectory)):
            error_Y = trajectory[min_idx + self.lookahead_long][1] - Y
            error_X = trajectory[min_idx + self.lookahead_long][0] - X
        else:
            error_Y = trajectory[len(trajectory) - 1][1] - Y
            error_X = trajectory[len(trajectory) - 1][0] - X        
        
        error_long = np.sqrt(error_X ** 2 + error_Y ** 2)  # computing distance to the waypoint at the end of the lookahead

        cum_error_long += error_long * delT  # adding current error to cumulative error
        d_error_long = (error_long - prev_error_long) / delT  # computing error derivative from current and previous error
        prev_error_long = error_long  # updating previous error
        
        F = Kp_long * error_long + Ki_long * cum_error_long + Kd_long * d_error_long  # longitudinal PID controller

        F = clamp(F, 0, 15736)  # setting F to be within the specified limits
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
