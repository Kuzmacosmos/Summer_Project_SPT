import numpy as np

def rot_matrix_3d(theta_x, theta_y, theta_z):
    """ Computes a general 3D rotation matrix from x-, y- and
    z-axial rotation angles. All inputs in degrees."""

    theta_x_rads = np.deg2rad(theta_x)
    theta_y_rads = np.deg2rad(theta_y)
    theta_z_rads = np.deg2rad(theta_z)

    R_x = np.array([[1, 0, 0],
                   [0, np.cos(theta_x_rads), -np.sin(theta_x_rads)],
                   [0, np.sin(theta_x_rads), np.cos(theta_x_rads)]])

    R_y = np.array([[np.cos(theta_y_rads), 0, np.sin(theta_y_rads)],
                   [0, 1, 0],
                   [-np.sin(theta_y_rads), 0, np.cos(theta_y_rads)]])

    R_z = np.array([[np.cos(theta_z_rads), -np.sin(theta_z_rads), 0],
                    [np.sin(theta_z_rads), np.cos(theta_z_rads), 0],
                    [0, 0, 1]])

    return np.matmul(np.matmul(R_z, R_y), R_x)

def rot_matrix_3d_vect(vector_in, theta_x, theta_y, theta_z):
    """ Computes the output vector based on the general 3D rotation matrix
    from axial rotation angles. Vector input in Cartesian,
    angle inputs in degrees.
    """

    return np.matmul(rot_matrix_3d(theta_x, theta_y, theta_z), vector_in)