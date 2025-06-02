import numpy as np

def refl_ray(ray_in: np.array, pl_norm: np.array) -> np.array:
    """ Computes the reflected ray on a surface based on the vectorial form of Law of Reflection.

        Referring to eq. (2.162), p. 69 of the textbook Springer handbook of lasers and optics; 2nd ed.

        https://doi.org/10.1007/978-3-642-19409-2

        a_2 = a_1 - 2 * (a_1 dot N) N

        Parameters:
             ray_in: incident ray vector (a_1)
             pl_norm: normal vector of a plane (N)

        Returns:
            reflected ray vector (a_2)
    """
    ray_out = ray_in - 2 * (np.dot(ray_in, pl_norm)) * pl_norm
    return ray_out