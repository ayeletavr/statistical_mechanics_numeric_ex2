import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----- initialize system -----

class System:
    """
    This class represents a system of 4 particles, containing all info nned to store.
    """

    def __init__(self, r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4):
        self.height, self.width = 1, 1
        self.r = r
        self.locations = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
        self.velocities = np.array([(vx1, vy1), (vx2, vy2), (vx3, vy3), (vx4, vy4)], dtype=float)
        self.i_particle_v = None
        self.j_particle_v = None
        self.i_particle = None
        self.j_particle = None
        self.v_tot_squared = np.sum(np.power(self.velocities[:, 0], 2)) + np.sum(np.power(self.velocities[:, 1], 2))
        self.wall_collision = False
        self.pair_collision = False
        self.particle_collided_with_wall = None

def init_system(r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4):
    """
    System constructor.
    :return: System object.
    """
    sys = System(r, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4)
    return sys

# ------- dynamics in time -------

def time_to_collision_wall(sys):
    """
    calculates time to collide in wall, for each particle
    :param sys: system
    :return: array of times to collision for each particle.
    """
    dtwalls = []
    for i in range(4):
        if sys.velocities[i][0] > 0: #vx is positive:
            dtwall_x = (1 - sys.r - sys.locations[i][0]) / sys.velocities[i][0]
        elif sys.velocities[i][0] < 0:
            dtwall_x = (sys.locations[i][0] - sys.r) / np.abs(sys.velocities[i][0])
        if sys.velocities[i][1] > 0: # vy is positive:
            dtwall_y = (1 - sys.r - sys.locations[i][1]) / sys.velocities[i][1]
        elif sys.velocities[i][1] < 0:
            dtwall_y = (sys.locations[i][1] - sys.r) / np.abs(sys.velocities[i][1])
        dtwalls.append(min(dtwall_x, dtwall_y))
    return dtwalls

def time_to_pair_collision(sys):
    """
    calculates time to collise with another particle, for eace particle.
    In addition, stores info about the minimal dt.
    :param sys: system
    :return: array of time to collision for each particle.
    """
    dtcolls = [] # 12 values - actually 6 (symmetry between (i, j) to (j, i).
    min_dtcoll = 10000000
    for i in range(4):
        for j in range(4):
            if i != j: # a particle cannot collide itself.
                delta_x = sys.locations[j][0] - sys.locations[i][0] # xj - xi
                delta_y = sys.locations[j][1] - sys.locations[i][1] # yj - yi
                # delta_l_squared = np.power(delta_x, 2) + np.power(delta_y, 2)
                delta_l = np.array([delta_x, delta_y])
                delta_l_squared = np.linalg.norm(delta_l) ** 2

                delta_vx = sys.velocities[j][0] - sys.velocities[i][0] #vxj - vxi
                delta_vy = sys.velocities[j][1] - sys.velocities[i][1] #vyj - vyi
                delta_v = np.array([delta_vx, delta_vy])
                delta_v_squared = np.linalg.norm(delta_v) ** 2
                # delta_v_squared = np.power(delta_vx, 2) + np.power(delta_vy, 2)

                s = np.dot(delta_vx, delta_x) + np.dot(delta_vy, delta_y)
                gamma = np.power(s, 2) - delta_v_squared * (delta_l_squared - 4 * np.power(sys.r, 2))
                # print("s: ", s)
                # print("gamma: ", gamma)
                # print("delta_x: ", delta_x, "delta_y: ", delta_y)
                # print("delta_vx: ", delta_vx, "delta_vy: ", delta_vy)
                # print("delta_l_squared: ", delta_l_squared)
                # print("delta_v_squared: ", delta_v_squared)
                # print("4r^2: ", 4 * np.power(sys.r, 2))

                if gamma > 0 and s < 0:
                    if (s + np.sqrt(gamma) <= 10 ** -3):
                        # print ("that's not good")
                        dtcoll = -(s + np.sqrt(gamma)) / delta_v_squared
                else:
                    dtcoll = 10000000
                dtcolls.append(dtcoll)

                if dtcoll < min_dtcoll: # save information for the min dtcoll in dtcolls.
                    min_dtcoll = dtcoll
                    e_x = delta_x / np.linalg.norm(delta_l)
                    e_y = delta_y / np.linalg.norm(delta_l)
                    s_v = np.dot(delta_vx, e_x) + np.dot(delta_vy, e_y)
                    new_vx_i = sys.velocities[i][0] + e_x * s_v
                    new_vx_j = sys.velocities[j][0] - e_x * s_v
                    new_vy_i = sys.velocities[i][1] + e_y * s_v
                    new_vy_j = sys.velocities[j][1] - e_y * s_v
                    sys.i_particle_v = (new_vx_i, new_vy_i)
                    sys.j_particle_v = (new_vx_j, new_vy_j)
                    sys.i_particle = i
                    sys.j_particle = j
                    # note: i need to save new velocities only for particles i and j that were part of the collision (min dists).
    # print(dtcolls)
    # print(len(dtcolls))
    return dtcolls

def find_dt(sys):
    """
    this function takes both dtcoll and dtwall times and finds the min of them.
    :param sys: system
    :return: dt
    """
    dtwalls = time_to_collision_wall(sys)
    dtcolls = time_to_pair_collision(sys)
    min_walls = np.min(dtwalls)
    sys.particle_collided_with_wall = dtwalls.index(min_walls)
    min_colls = np.min(dtcolls)
    # print(dtcolls)
    # print("min wall: ", min_walls, "min coll: ", min_colls)
    if min_walls < min_colls:
        dt = min_walls
        sys.wall_collision = True
    else:
        dt = min_colls
        sys.pair_collision = True
    # print(dtcolls)
    # print(dtwalls)
    return dt

def update_system(sys, dt):
    """
    this function updates locations and velocities of all particles, after dt.
    :param sys: system
    :param dt: time to nect collision.
    """
    for i in range(4): # update locations
        new_x = sys.locations[i][0] + sys.velocities[i][0] * dt
        new_y = sys.locations[i][1] + sys.velocities[i][1] * dt
        # sys.locations[i][0] = new_x
        # sys.locations[i][1] = ney_y
        sys.locations[i] = (new_x, new_y)
        # print("new location of ", i, " :", sys.locations[i])
    if sys.wall_collision:
        # print("wall collusion")
        # print(sys.locations[sys.particle_collided_with_wall])
        # if sys.locations[sys.particle_collided_with_wall][0] in [sys.r, 1 - sys.r]:
        if sys.locations[sys.particle_collided_with_wall][0] <= 0 + sys.r or sys.locations[sys.particle_collided_with_wall][0] >= 1 - sys.r:
            # print("hi")
            sys.velocities[sys.particle_collided_with_wall] = (sys.velocities[sys.particle_collided_with_wall][0] * -1, sys.velocities[sys.particle_collided_with_wall][1])
        else:
        # elif sys.locations[sys.particle_collided_with_wall][1] <= 0 + sys.r or sys.locations[sys.particle_collided_with_wall][1] >= 1 - sys.r:
            # print("hhh")
            sys.velocities[sys.particle_collided_with_wall] = (sys.velocities[sys.particle_collided_with_wall][0], sys.velocities[sys.particle_collided_with_wall][1] * -1)
    elif sys.pair_collision:
        sys.velocities[sys.i_particle] = (sys.i_particle_v[0], sys.i_particle_v[1])
        # print("new v of i: ", sys.velocities[sys.i_particle])
        sys.velocities[sys.j_particle] = (sys.j_particle_v[0], sys.j_particle_v[1])
        # sys.velocities[sys.i_particle][0], sys.velocities[sys.i_particle][1] = sys.i_particle_v[0], sys.i_particle_v[1]
        # sys.velocities[sys.j_particle][0], sys.velocities[sys.j_particle][1] = sys.j_particle_v[0], sys.j_particle_v[1]
    else:
        print("not supposed to happen")

# -------------------- Tracking particles ----------------------

# x_axis = np.linspace(0, 1, 10)
# y_axis = np.linspace(0, 1, 10)
# x_y_box = np.meshgrid(x_axis, y_axis, sparse=True)
def get_v_axis(sys):
    return np.linspace(sys.v_tot_squared * -1, sys.v_tot_squared, 200)

def simulate(sys, dtstore, N):
    collides_ctr = 0
    t = 0
    p1_locations = []
    while collides_ctr < N:
        sys.wall_collision = False
        sys.pair_collision = False
        sys.particle_collided_with_wall = None
        sys.i_particle_v = None
        sys.j_particle_v = None
        sys.i_particle = None
        sys.j_particle = None
        dt = find_dt(sys)

        if t + dt >= dtstore:
            update_system(sys, dt)
            print("step: ", collides_ctr)
            # print("updated locations: ", sys.locations)
            # print("updated velocities: ", sys.velocities)
            p1_locations.append((sys.locations[0][0], sys.locations[0][1])) # append (x,y) after dt for qa1
            t = 0
            collides_ctr += 1
        else:
            t += dt
    return p1_locations

# ---------- Computes -----------
def reference_compute(r):
    # velocities = [(0.21, 0.12), (0.71, 0.18), (-0.23, -0.79), (0.78, 0.34583)]
    sys = init_system(r, 0.21, 0.12, 0.71, 0.18, -0.23, -0.79, 0.78, 0.34583)
    N = 10 ** 7
    dtstore = 1.0
    p1_arr = simulate(sys, dtstore, N)
    return p1_arr
# run base case with r = 0.15

def comparing_computes():
    r_vals = np.arange(0.1, 0.23, 0.01)
    for r in r_vals:
        reference_compute(r)

# ------- results output ------
#  a - 1
def qa1():
    x_axis = np.linspace(0, 1, 10)
    y_axis = np.linspace(0, 1, 10)
    x_bounds, y_bounds = np.meshgrid(x_axis, y_axis, sparse=True)
    x, y = zip(*reference_compute(r=0.15))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=10)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Heatmap of one particle')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    qa1()







