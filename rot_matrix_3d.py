import numpy as np

def rot_matrix_3d(roll, pitch, yaw):
    """ Computes a general 3D rotation matrix from roll (x), pitch (y) and
    yaw (z) angles. All inputs in degrees."""

    roll_rads = np.deg2rad(roll)
    pitch_rads = np.deg2rad(pitch)
    yaw_rads = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0],
                   [0, np.cos(roll_rads), -np.sin(roll_rads)],
                   [0, np.sin(roll_rads), np.cos(roll_rads)]])

    R_y = np.array([[np.cos(pitch_rads), 0, np.sin(pitch_rads)],
                   [0, 1, 0],
                   [-np.sin(pitch_rads), 0, np.cos(pitch_rads)]])

    R_z = np.array([[np.cos(yaw_rads), -np.sin(yaw_rads), 0],
                    [np.sin(yaw_rads), np.cos(yaw_rads), 0],
                    [0, 0, 1]])

    return np.matmul(np.matmul(R_z, R_y), R_x)