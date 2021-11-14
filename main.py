import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

# ----- initialize system -----

class System:

    def __init__(self, r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4):
        self.height, self.width = 1, 1
        self.r = r
        self.locations = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
        self.velocities = [(vx1, vy1), (vx2, vy2), (vx3, vy3), (vx4, vy4)]

def init_system(r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4):
    sys = System(r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4)
    return sys

# ------- dynamics in time -------

def time_to_collision_wall(sys):
    dtwalls = []
    for i in range(4):
        if sys.velocities[i][0] > 0: #vx is positive:
            dtwall_x = (1 - sys.r - sys.locations[i][0]) / sys.velocities[i][0]
        else:
            dtwall_x = (sys.locations[i][0] - sys.r) / np.abs(sys.velocities[i][0])
        if sys.velocities[i][1] > 0: # vy is positive:
            dtwall_y = (1 - sys.r - sys.locations[i][1]) / sys.velocities[i][1]
        else:
            dtwall_y = (sys.locations[i][1] - sys.r) / np.abs(sys.velocities[i][1])
        dtwalls.append(np.max(dtwall_x, dtwall_y))
    return dtwalls

def time_to_pair_collision(sys):
    dtcolls = [] #6 values.
    new_v = [[]]
    for i in range(0,4):
        for j in range (0,4):
            if i == j:
                pass # a particle cannot collide itself.
            else:

                delta_x = sys.locations[j][0] - sys.locations[i][0] #xj - xi
                delta_y = sys.locations[j][1] - sys.locations[i][1] # yj - yi
                delta_l_squared = delta_x ^ 2 + delta_y ^ 2

                delta_vx = sys.velocities[j][0] - sys.velocities[i][0]
                delta_vy = sys.velocities[j][1] - sys.velocities[i][1]
                delta_v_squared = delta_vx ^ 2 + delta_vy ^ 2

                s = np.dot(delta_vx, delta_x) + np.dot(delta_vy, delta_y)
                gamma = s ^ 2 - delta_v_squared * (delta_l_squared - 4 * sys.r ^ 2)

                if gamma > 0 and s < 0:
                    dtcoll = (s + np.sqrt(gamma)) / delta_v_squared
                else:
                    dtcoll = 10000000
                dtcolls.append(dtcoll)

                e_x = delta_x / np.sqrt(delta_l_squared)
                e_y = delta_y / np.sqrt(delta_l_squared)
                s_v = np.dot(delta_vx, e_x) + np.dot(delta_vy, e_y)
                new_vx_i = sys.velocities[i][0] + e_x * s_v
                new_vx_j = sys.velocities[j][0] + e_x * s_v
                new_vy_i = sys.velocities[i][1] + e_y * s_v
                new_vy_j = sys.velocities[j][1] + e_y * s_v
                # note: i need to save new velocities only for particles i and j that were part of the collision (min dists).
    return dtcolls

def find_dt_and_update_system(sys):
    wall_collision = False
    pair_collision = False
    dtwalls = time_to_collision_wall(sys)
    dtcolls = time_to_pair_collision(sys)
    min_walls = np.min(dtwalls)
    min_colls = np.min(dtcolls)
    if min_walls < min_colls:
        dt = min_walls
        wall_collision = True
    else:
        dt = min_colls
        pair_collision = True
    for i in range(4): # update locations
        new_x = sys.locations[i][0] + sys.velocities[i][0] * dt
        ney_y = sys.locations[i][1] + sys.velocities[i][1] * dt
        sys.locations[i][0] = new_x
        sys.locations[i][1] = ney_y
        if wall_collision:
            if sys.locations[i][0] == 0 or sys.locations[i][0] == 1:
                sys.locations[i][0] *= -1
            else:
                sys.locations[i][1] += -1
        elif pair_collision:

