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
H = np.array([224.5, -612.5, 201])

# Mirror geometry
a = 140           # distance from H to each mirror center (mm)
mirror_diam = 25  # mirror diameter (mm)
phi = np.deg2rad(7)   # local tilting of mirrors (rad)

# Belt angles (radians)
beta = np.deg2rad(-99)   # pitch belt rotation (rad)
gamma = np.deg2rad(-16)  # yaw belt rotation (rad)

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
n_loc = [rot_x(-phi) @ z_hat, rot_x(+phi) @ z_hat] # n_m1^loc, n_m2^loc

# Step 4
n_mi_0 = [M @ n for n in n_loc] # Eq. 5.1

# Step 5
u_y = M @ np.array([0,1,0]) # Eq. 6.1
n_mi_1 = [rodrigues(n_mi_0[i], u_y, beta/2) for i, u_y in enumerate([u_y, u_y])]

# Step 6
n_h_final = n_h.copy()
n_mi_final = [rodrigues(n_mi_1[i], n_h, gamma) for i in range(2)] # final mirror normals


# Step 7
mi_1 = [H - a * u_y, H + a * u_y]
mi_2 = mi_1.copy() # positions unchanged


mi_final = [H + rodrigues(mi_1[i]-H, n_h, gamma) for i in range(2)]

# Solar ray from solar angle definitions
alt, azi = solar_angles(LAT, LON, TRIAL_UTC)
incoming = mod_sph_to_cart(alt, 180)
# Above, we set azimuth=180, to simulate the sunlight from south, shining
# northward.

# Reflection and intersection for each mirror
endpoints = []
refl_dirs = []
for center, normal in zip(mi_final, n_mi_final):
    refl_dir = refl_ray(incoming, normal)
    refl_dirs.append(refl_dir)
    t = (RECEIVER_POS[2] - center[2]) / refl_dir[2]
    endpoints.append(center + t * refl_dir)
endpoints = np.array(endpoints)

print(incoming)
print(n_loc)
print(n_mi_0)
print(n_mi_1)
print(n_mi_final)

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

# Build the figure (the following code suggested by ChatGPT)
fig = go.Figure()

# Mirror disks
for C, N in zip(mi_final, n_mi_final):
    # create disk in plane
    r = mirror_diam/2
    n_pts = 50
    angles = np.linspace(0, 2*np.pi, n_pts)
    # find orthonormal basis (u,v) for the plane
    # pick arbitrary vector not parallel to N
    arb = np.array([0,0,1]) if abs(N[2])<0.9 else np.array([0,1,0])
    u = np.cross(N, arb); u /= np.linalg.norm(u)
    v = np.cross(N, u)
    circle_pts = np.array([C + r*(np.cos(t)*u + np.sin(t)*v) for t in angles])
    # vertices: center + circle
    verts = np.vstack([C, circle_pts])
    x, y, z = verts.T
    # triangles: fan from center
    I = np.zeros(n_pts, dtype=int)
    J = np.arange(1, n_pts+1)
    K = np.roll(J, -1)
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                            i=I, j=J, k=K,
                            opacity=0.5,
                            color='lightblue',
                            name='Mirror',
                            showscale=False))

# Incoming rays
ray_len = 300
for C in mi_final:
    start = C - incoming * ray_len
    fig.add_trace(go.Scatter3d(x=[*start[:1],*C[:1]], y=[start[1],C[1]], z=[start[2],C[2]],
                               mode='lines', line=dict(color='orange', width=4),
                               name='Incoming Ray', showlegend=False))

# Reflected rays
for C, E in zip(mi_final, endpoints):
    fig.add_trace(go.Scatter3d(x=[C[0], E[0]], y=[C[1], E[1]], z=[C[2], E[2]],
                               mode='lines', line=dict(color='red', width=4),
                               name='Reflected Ray', showlegend=False))

# Normals as arrows
arrow_len = 50
# helio-unit normal at H
starts = [H] + mi_final
normals = [n_h] + n_mi_final
for start, N in zip(starts, normals):
    tip = start + N * arrow_len
    fig.add_trace(go.Scatter3d(x=[start[0], tip[0]],
                               y=[start[1], tip[1]],
                               z=[start[2], tip[2]],
                               mode='lines',
                               line=dict(color='green', width=6),
                               name='Normal', showlegend=False))
    # arrowhead
    # pick two perpendicular vectors for wings
    arb = np.array([0,0,1]) if abs(N[2])<0.9 else np.array([0,1,0])
    b1 = np.cross(N, arb); b1 /= np.linalg.norm(b1)
    b2 = np.cross(N, b1)
    head_len = arrow_len * 0.2
    wing1 = tip - N*head_len + b1*head_len*0.5
    wing2 = tip - N*head_len - b1*head_len*0.5
    fig.add_trace(go.Scatter3d(x=[tip[0], wing1[0]], y=[tip[1], wing1[1]], z=[tip[2], wing1[2]],
                               mode='lines', line=dict(color='green', width=6), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[tip[0], wing2[0]], y=[tip[1], wing2[1]], z=[tip[2], wing2[2]],
                               mode='lines', line=dict(color='green', width=6), showlegend=False))

# Receiver plane
xmin, xmax = -300, 300
ymin, ymax = -300, 300
xx = np.array([[xmin, xmax], [xmin, xmax]])
yy = np.array([[ymin, ymin], [ymax, ymax]])
zz = np.full_like(xx, RECEIVER_POS[2])
fig.add_trace(go.Surface(x=xx, y=yy, z=zz,
                         showscale=False, opacity=0.2,
                         colorscale=[[0, 'gray'], [1, 'gray']],
                         name='Receiver Plane'))

# Hit points and receiver center
fig.add_trace(go.Scatter3d(x=endpoints[:,0], y=endpoints[:,1], z=endpoints[:,2],
                           mode='markers', marker=dict(color='magenta', size=6),
                           name='Hit Points'))
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[RECEIVER_POS[2]],
                           mode='markers', marker=dict(color='black', symbol='cross', size=8),
                           name='Receiver Center'))

# Layout tweaks
fig.update_layout(scene=dict(aspectmode='data',
                             xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
                  legend=dict(itemsizing='constant'),
                  margin=dict(l=0, r=0, t=0, b=0))

fig.show()