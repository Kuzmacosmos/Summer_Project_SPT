import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# Ensure plotly opens in the default browser
pio.renderers.default = "browser"

# Parameters
# Receiver position R = (0,0,z_r)
RECEIVER_POS = np.array([0.0, 0.0, 771.6])  # mm

# Heliostat support base center H = (x_h, y_h, z_h)
H = np.array([222.5, -612.5, 201])

# Mirror geometry
a = 140           # distance from H to each mirror center (mm)
mirror_diam = 25  # mirror diameter (mm)
phi = np.deg2rad(0.0)   # local tilting of mirrors (rad)

# Belt angles (radians)
beta = np.deg2rad(-140)   # pitch belt rotation (rad)
gamma = np.deg2rad(-33)  # yaw belt rotation (rad)

# Solar observation
LAT, LON, TZ_OFF = 51.49, -0.177, 0
TRIAL_LOCAL = dt.datetime(2025, 3, 20, 12, 0)
TRIAL_UTC = TRIAL_LOCAL - dt.timedelta(hours=TZ_OFF)

# Helpers
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rodrigues(v, k, angle):
    """Rotate vector v about axis k by angle (rad)."""
    k = k/np.linalg.norm(k)
    return (v*np.cos(angle) + np.cross(k,v)*np.sin(angle)
            + k*(np.dot(k,v))*(1-np.cos(angle)))

from sunPos import solar_angles
from sph_to_cart import mod_sph_to_cart
from refl_ray import refl_ray

# Step 1
HR = RECEIVER_POS - H
n_h = HR / np.linalg.norm(HR)

# Step 2
D = np.linalg.norm(HR) # dist from helio to receiver
theta_y = np.arccos((RECEIVER_POS[2]-H[2]) / D)
theta_z = np.arctan2(-H[1], -H[0])
M = rot_z(theta_z) @ rot_y(theta_y)

# Step 3
z_hat = np.array([0,0,1]) # standard upright vector // global z-axis
n_loc = [rot_y(-phi) @ z_hat, rot_y(+phi) @ z_hat] # n_m1^loc, n_m2^loc

# Step 4
n_mi_0 = [M @ n for n in n_loc] # Eq. 5.1

# Step 5
u_x = M @ np.array([1,0,0]) # Eq. 6.1
n_mi_1 = [rodrigues(n_mi_0[i], u_x, beta/2) for i, u_x in enumerate([u_x, u_x])]

# Step 6
n_h_final = n_h.copy()
n_mi_final = [rodrigues(n_mi_1[i], n_h, gamma) for i in range(2)] # final mirror normals


# Step 7
mi_0 = [H - [a, 0, 0], H + [a, 0, 0]]
mi_1 = [H - a * u_x, H + a * u_x]
mi_2 = mi_1.copy() # positions unchanged


mi_final = [H + rodrigues(mi_1[i]-H, n_h, gamma) for i in range(2)]

# -- Solar ray --
alt, azi = solar_angles(LAT, LON, TRIAL_UTC)
incoming = mod_sph_to_cart(alt, 0)

# -- Reflection and intersection for each mirror --
endpoints = []
for i, center in enumerate(mi_final):
    refl_dir = refl_ray(incoming, n_mi_final[i])
    t = (RECEIVER_POS[2] - center[2]) / refl_dir[2]
    endpoints.append(center + t * refl_dir)
endpoints = np.array(endpoints)


plt.figure()
plt.scatter(endpoints[:,0], endpoints[:,1], label='Reflected hits')
plt.scatter([0],[0], c='r', marker='+', s=100, label='Receiver center')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
# plt.xlim(-300,300)
# plt.ylim(-200,200)
plt.title('Reflection points on receiver plane')
plt.legend()
plt.show()