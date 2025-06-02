import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# Ensure plotly opens in the default browser
pio.renderers.default = "browser"

from sunPos import solar_angles, format_lat_lon, format_timezone
from sph_to_cart import mod_sph_to_cart
from compute_plane_norm import compute_plane_normal
from refl_ray import refl_ray

""" Adapted from Springer Handbook of Lasers and Optics (2nd edition), Section 2.4.1.
    https://doi.org/10.1007/978-3-642-19409-2
    Code refactored by ChatGPT.
"""

# Initial parameters
LAT_DEG, LON_DEG, TZ_OFFSET = 51.4991, -0.1789, 1
TRIAL_LOCAL = dt.datetime(2025, 6, 2, 15, 0)  # local time
TRIAL_UTC = TRIAL_LOCAL - dt.timedelta(hours=TZ_OFFSET)  # convert to UTC

RECEIVER_POS = np.array([0.0, 0.0, 10.0])

# Heliostat grid
GRID_SIZE = 10
Z_MIRROR = 3.0
X_COORDS = np.linspace(-10, 10, GRID_SIZE)
Y_COORDS = np.linspace(-10, 10, GRID_SIZE)
MIRROR_CENTERS = np.array([[x, y, Z_MIRROR] for x in X_COORDS for y in Y_COORDS])

# Compute solar angles and incoming ray
lat_str, lon_str = format_lat_lon(LAT_DEG, LON_DEG)
tz_str = format_timezone(TZ_OFFSET)
altitude, azimuth = solar_angles(LAT_DEG, LON_DEG, TRIAL_UTC)
incoming_ray = mod_sph_to_cart(altitude, azimuth)

print(f"Observation point at ({lat_str}, {lon_str}):")
print(f"At local time {TRIAL_LOCAL.strftime('%Y-%m-%d %H:%M')} (UTC+"
      f"{TZ_OFFSET:02d}:00) <- {TRIAL_UTC.strftime('%Y-%m-%d %H:%M')} "
      f"(UTC):")
print(f" - Solar altitude: {altitude:.2f}°; azimuth: {azimuth:.2f}°")

# Vector geometry
sun_points = []
mirror_normals = []
reflected_dirs = []
endpoints = []

for center in MIRROR_CENTERS:
    sun_pt = center - 3 * incoming_ray # Choose s0 = -3 to make the rays visible on the plot
    sun_points.append(sun_pt)

    normal_vec = compute_plane_normal(incoming_ray, center, RECEIVER_POS)
    mirror_normals.append(normal_vec)

    refl_dir = refl_ray(incoming_ray, normal_vec)
    reflected_dirs.append(refl_dir)

    t = (RECEIVER_POS[2] - center[2]) / refl_dir[2]
    endpoint = center + t * refl_dir
    endpoints.append(endpoint)

sun_points      = np.array(sun_points)
mirror_normals  = np.array(mirror_normals)
reflected_dirs  = np.array(reflected_dirs)
endpoints       = np.array(endpoints)


# Construct a plotly interactive plot
fig = go.Figure()

# Mirrors
fig.add_trace(go.Scatter3d(
    x=MIRROR_CENTERS[:, 0],
    y=MIRROR_CENTERS[:, 1],
    z=MIRROR_CENTERS[:, 2],
    mode='markers',
    marker=dict(size=4, color='red'),
    name='Mirror Centers'
))

# Receiver
fig.add_trace(go.Scatter3d(
    x=[RECEIVER_POS[0]],
    y=[RECEIVER_POS[1]],
    z=[RECEIVER_POS[2]],
    mode='markers+text',
    marker=dict(size=6, color='black'),
    text=['Receiver'],
    textposition='top center',
    name='Receiver'
))

# Incidents/Normals/Reflections
for i, center in enumerate(MIRROR_CENTERS):
    sun_pt     = sun_points[i]
    normal_vec = mirror_normals[i]
    refl_dir   = reflected_dirs[i]
    end_pt     = endpoints[i]

    # Incident ray, from (arbitrary) sun point to center
    fig.add_trace(go.Scatter3d(
        x=[sun_pt[0], center[0]],
        y=[sun_pt[1], center[1]],
        z=[sun_pt[2], center[2]],
        mode='lines',
        line=dict(color='orange', width=2),
        showlegend=(i == 0),
        name='Incident Ray'
    ))

    # Mirror normal
    normal_end = center + normal_vec * 1.0
    fig.add_trace(go.Scatter3d(
        x=[center[0], normal_end[0]],
        y=[center[1], normal_end[1]],
        z=[center[2], normal_end[2]],
        mode='lines',
        line=dict(color='purple', width=2),
        showlegend=(i == 0),
        name='Mirror Normal'
    ))

    # Reflected ray, from center to end point
    fig.add_trace(go.Scatter3d(
        x=[center[0], end_pt[0]],
        y=[center[1], end_pt[1]],
        z=[center[2], end_pt[2]],
        mode='lines',
        line=dict(color='green', width=2),
        showlegend=(i == 0),
        name='Reflected Ray'
    ))

fig.update_layout(
    title='Mirror Array: Incident/Reflected Rays & Normals',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    )
)

fig.show()

# Intersection points of the reflected rays on receiver plane z=10
plt.figure(figsize=(8,8))
plt.scatter(endpoints[:, 0], endpoints[:, 1])
plt.xlabel('X (a.u.)')
plt.ylabel('Y (a.u.)')
plt.xlim(-10e-15, 10e-15)
plt.ylim(-10e-15, 10e-15)
plt.title('Intersection points of the reflected rays on receiver plane (x, y, 10)')
plt.show()

