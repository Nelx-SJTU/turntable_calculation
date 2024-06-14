import numpy as np
from scipy.optimize import least_squares
from sklearn.decomposition import PCA


class FixedSizeArray:
    def __init__(self, max_len=100, data_size=(0, 3)):
        self.max_len = max_len
        self.data_size = data_size
        self.data = np.empty(self.data_size)

    def append(self, value):
        if self.data.size < self.max_len*self.data_size[1]:
            self.data = np.append(self.data, value, axis=0)
        else:
            self.data = np.append(self.data[1:], value, axis=0)

    def get_array(self):
        return self.data


def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


# (alpha, beta, gamma) = (roll, pitch, yaw) = (x, y, z)
def create_rotation_matrix(alpha, beta, gamma):
    r11 = np.cos(gamma)*np.cos(beta)
    r12 = -np.sin(gamma)*np.cos(alpha)+np.cos(gamma)*np.sin(beta)*np.sin(alpha)
    r13 = np.sin(gamma)*np.sin(alpha)+np.cos(gamma)*np.sin(beta)*np.cos(alpha)
    r21 = np.sin(gamma)*np.cos(beta)
    r22 = np.cos(gamma)*np.cos(alpha)+np.sin(gamma)*np.sin(beta)*np.sin(alpha)
    r23 = -np.cos(gamma)*np.sin(alpha)+np.sin(gamma)*np.sin(beta)*np.cos(alpha)
    r31 = -np.sin(beta)
    r32 = np.cos(beta)*np.sin(alpha)
    r33 = np.cos(beta)*np.cos(alpha)
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


# # Define the circular fitting error function
# def circle_error(params, points):
#     center, radius = params[:3], params[3]
#     return np.linalg.norm(points - center, axis=1) - radius


# # Function of circle to be optimized
# def circle_3d(params, points):
#     center = params[:3]
#     normal = params[3:6]
#     radius = params[6]
#
#     # Normalize the normal vector
#     normal = normal / np.linalg.norm(normal)
#
#     # Vector from center to each point
#     v = points - center
#
#     # Project vectors onto plane defined by normal vector
#     v_proj = v - np.outer(np.dot(v, normal), normal)
#
#     # Calculate distances from the center of the circle to the points
#     distances = np.linalg.norm(v_proj, axis=1)
#
#     # Residuals are the differences from the desired radius
#     residuals = distances - radius
#
#     return residuals


# def fit_circle_3d(points):
#     # Initial guess for the parameters
#     center_initial = np.mean(points, axis=0)
#     normal_initial = np.array([0, 0, 1])
#     radius_initial = np.mean(np.linalg.norm(points - center_initial, axis=1))
#
#     initial_params = np.hstack((center_initial, normal_initial, radius_initial))
#
#     # Perform least squares fitting
#     result = least_squares(circle_3d, initial_params, args=(points,))
#
#     # Extract the fitted parameters
#     center_fitted = result.x[:3]
#     normal_fitted = result.x[3:6] / np.linalg.norm(result.x[3:6])
#     radius_fitted = result.x[6]
#
#     return center_fitted, radius_fitted, normal_fitted


# def fit_circle_3d_old(points):
#     # Calculate the centroid of the points
#     centroid = np.mean(points, axis=0)
#
#     # Calculate the initial radius as the distance from centroid to the first point
#     radius_initial = np.mean(np.linalg.norm(points - centroid, axis=1))
#
#     # Initial guess for the parameters
#     center_initial = centroid
#
#     initial_params = np.hstack((center_initial, radius_initial))
#
#     # Perform least squares fitting
#     result = least_squares(circular_residual_equation, initial_params, args=(points,))
#
#     # Extract the fitted parameters
#     center_fitted = result.x[:3]
#     radius_fitted = result.x[3]
#
#     # Calculate the vector points
#     v_array = np.empty((0, 3))
#     for i, point in enumerate(points):
#         vec_to_center = point - center_fitted
#         v_array = np.append(v_array, np.array([vec_to_center]), axis=0)
#
#     # Calculate fitted normal (rotation axis)
#     n_array = np.empty((0, 3))
#     for i in range(int(len(v_array)/2)):
#         vec_normal = np.cross(v_array[i], v_array[int(len(v_array)/2 + i)])
#         n_array = np.append(n_array, np.array([vec_normal]) / np.linalg.norm(vec_normal), axis=0)
#     print('n_array', n_array)
#     normal_fitted = np.mean(n_array, axis=0)
#     normal_fitted = normal_fitted / np.linalg.norm(normal_fitted)
#
#     return center_fitted, radius_fitted, normal_fitted


def fit_circle_3d(points):
    # STEP 1 : Calculate the normal_fitted
    # Calculate the vector points
    v_array = np.empty((0, 3))
    for i in range(int(len(points) / 2)):
        vec_between_points = points[i] - points[int(len(points) / 2 + i)]
        v_array = np.append(v_array, np.array([vec_between_points]), axis=0)

    # Calculate fitted normal (rotation axis)
    n_array = np.empty((0, 3))
    for i in range(len(v_array) - 1):
        vec_normal = np.cross(v_array[i], v_array[i + 1])
        n_array = np.append(n_array, np.array([vec_normal]), axis=0)
    normal_fitted = np.mean(n_array, axis=0)
    normal_fitted = - normal_fitted / np.linalg.norm(normal_fitted)  # + or - depends on camera position

    # STEP 2 : Project center on the rotation axis
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the initial radius as the distance from centroid to the first point
    radius_initial = np.mean(np.linalg.norm(points - centroid, axis=1))

    # Initial guess for the parameters
    center_initial = centroid

    initial_params = np.hstack((center_initial, radius_initial))

    # Perform least squares fitting
    def circular_residual_equation_1(params, points):
        center, radius = params[:3], params[3]
        residual = np.linalg.norm(points - center, axis=1) - radius
        return residual

    result = least_squares(circular_residual_equation_1, initial_params, args=(points,))

    # Extract the fitted parameters
    center_estimated = result.x[:3]
    # radius_estimated = result.x[3]

    # STEP 3 : Calculate center_fitted, radius_fitted
    def circular_residual_equation_2(t, points, C, v):
        # Calculate the point on the line C + t*v
        P = C + t * v
        # Calculate the distance from P to each point in points
        distances = np.linalg.norm(points - P, axis=1)
        # Return the mean distance
        return np.mean(distances)

    t_initial = 0.0

    # Perform least squares optimization to find the optimal t
    result = least_squares(circular_residual_equation_2, t_initial, args=(points, center_estimated, normal_fitted))

    # Calculate the optimal point
    t_optimal = result.x[0]
    center_fitted = center_estimated + t_optimal * normal_fitted
    radius_array = np.array([np.linalg.norm(point - center_fitted) for point in points])
    radius_fitted = np.mean(radius_array)

    return center_fitted, radius_fitted, normal_fitted


def plot_circle(ax, center, radius, normal, num_points=100):
    # Creating a unit circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])

    # Constructing rotation matrices using orthogonal normal vectors
    normal = normal / np.linalg.norm(normal)
    if (normal == [0, 0, 1]).all():
        R = np.eye(3)
    else:
        v = np.cross([0, 0, 1], normal)
        c = np.dot([0, 0, 1], normal)
        s = np.linalg.norm(v)
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s ** 2))

    # Rotate and translate the circle
    circle_rot = R @ circle
    circle_rot[0, :] += center[0]
    circle_rot[1, :] += center[1]
    circle_rot[2, :] += center[2]

    # Drawing Circles
    ax.plot(circle_rot[0, :], circle_rot[1, :], circle_rot[2, :], 'r-')


