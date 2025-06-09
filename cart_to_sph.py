import numpy as np

def mod_cart_to_sph(vec: np.array) -> np.array:
    """ Modified inverse: recover (alt, azi) in degrees from a Cartesian
    unit vector.

        Reverse by:
            x = -cos(alt) * sin(azi)
            y = -cos(alt) * cos(azi)
            z = -sin(alt)

        Output in degrees.
    """
    x, y, z = vec
    rho = np.linalg.norm(vec)

    x_u = x / rho
    y_u = y / rho
    z_u = z / rho

    # altitude: sin(alt) = -z_u
    alt_rads = np.arcsin(-z_u)
    alt = np.rad2deg(alt_rads)

    # azimuth: tan(azi) = (sin(azi)/cos(azi)) = (-x_u)/(-y_u)
    azi_rads = np.arctan2(-x_u, -y_u)
    azi = np.rad2deg(azi_rads) % 360

    return np.array([alt, azi])
