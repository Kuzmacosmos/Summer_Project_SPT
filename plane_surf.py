import numpy as np

def plane_surf(init_pos: np.array, dir_vec: np.array, pl_norm: np.array, center: np.array) -> np.array:
    """ Computes a position of a plane of a given initial position, direction vector,
        plane's center point and plane's normal.

        Referring to eq. (2.145), p. 66 of the textbook Springer handbook of lasers and optics; 2nd ed.
        https://doi.org/10.1007/978-3-642-19409-2

        s_0 = (C - p) dot n_z / (a dot n_z)
        r = p + s_0 * a

        Parameters:
             init_pos (array): initial position vector (p)
             dir_vec (array): direction vector (a)
             pl_norm (array): plane normal vector (n_z)
             center (array): center point (C)
        Returns:
            Output position vector (r)
    """
    s_0 = np.dot((center - init_pos), pl_norm) / np.dot(dir_vec, pl_norm)
    pos_out = init_pos + s_0 * dir_vec
    return pos_out


