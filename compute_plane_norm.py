import numpy as np

def compute_plane_normal(ray_in: np.array,
                      mirror_center: np.array,
                      receiver_pos: np.array) -> np.array:
    """
    Calculate the unit normal vector of a (flat) mirror such that an incident ray
    ray_in, striking at mirror_center, will be reflected toward receiver_pos.

    Parameters
        ray_in : np.array, a_1
            Unit‐vector direction of the incoming solar ray (pointing into the mirror).
            For instance, from mod_sph_to_cart(alt, azi).
        mirror_center : np.array, C
            The 3‐vector (x0, y0, z0) of the point on the mirror where the ray hits.
            (By assumption, the ray actually does hit that exact point.)
        receiver_pos : np.array
            The 3‐vector (0, 0, z_r) of the receiver’s location.
    Returns
        normal : np.ndarray, n_z or N
            The unit‐length normal of the plane (mirror).  This n is chosen so that
            (1) n dot ray_in < 0 (i.e. n “faces” the incoming ray)
            (2) reflecting ray_in about n exactly sends it toward receiver_pos.

    Notes
        - Law of reflection (vector form):  r_out = r_in - 2 (r_in dot n) n.
          We solve a1 - a2 = 2 (a1 dot n) n  => n || (a1 - a2).
        - We pick the sign of n so that a1 dot n < 0.
    """

    v = receiver_pos - mirror_center
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("mirror_center and receiver_pos cannot coincide.")
    a2 = v / norm_v

    # form the (unnormalized) candidate for normal:  a1 - a2
    delta = ray_in - a2
    norm_delta = np.linalg.norm(delta)
    if norm_delta == 0:
        raise ValueError("Incoming direction equals desired outgoing direction; "
                         "degenerate (no unique mirror normal).")
    delta_unit = delta / norm_delta


    if np.dot(ray_in, delta_unit) < 0:
        n = delta_unit.copy()
    else:
        n = -delta_unit.copy()

    return n
