from common import np


def evaluate_at_gauss_points(shape_func, shape_func_b, u):
    u_G_domain = np.dot(shape_func, u)
    u_G_boundary = np.dot(shape_func_b, u)  # all inputs are np array

    return u_G_domain, u_G_boundary
