"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    # print("In FK_dh, dh_params = ", dh_params)
    i = 0
    homgen_0_1 = get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3])
    i = 1
    homgen_1_2 = get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3])
    i = 2
    homgen_2_3 = get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3])
    i = 3
    homgen_3_4 = get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3])
    i = 4
    homgen_4_5 = get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3])
    # i = 1
    # homgen_1_2 = np.array([[np.cos(dh_params[i,3]), -np.sin(dh_params[i,3]) * np.cos(dh_params[i,1]), np.sin(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.cos(dh_params[i,3])],
    #                     [np.sin(dh_params[i,3]), np.cos(dh_params[i,3]) * np.cos(dh_params[i,1]), -np.cos(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.sin(dh_params[i,3])],
    #                     [0, np.sin(dh_params[i,1]), np.cos(dh_params[i,1]), dh_params[i,2]],
    #                     [0, 0, 0, 1]])
    # i = 2
    # homgen_2_3 = np.array([[np.cos(dh_params[i,3]), -np.sin(dh_params[i,3]) * np.cos(dh_params[i,1]), np.sin(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.cos(dh_params[i,3])],
    #                     [np.sin(dh_params[i,3]), np.cos(dh_params[i,3]) * np.cos(dh_params[i,1]), -np.cos(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.sin(dh_params[i,3])],
    #                     [0, np.sin(dh_params[i,1]), np.cos(dh_params[i,1]), dh_params[i,2]],
    #                     [0, 0, 0, 1]])
    # i = 3
    # homgen_3_4 = np.array([[np.cos(dh_params[i,3]), -np.sin(dh_params[i,3]) * np.cos(dh_params[i,1]), np.sin(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.cos(dh_params[i,3])],
    #                     [np.sin(dh_params[i,3]), np.cos(dh_params[i,3]) * np.cos(dh_params[i,1]), -np.cos(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.sin(dh_params[i,3])],
    #                     [0, np.sin(dh_params[i,1]), np.cos(dh_params[i,1]), dh_params[i,2]],
    #                     [0, 0, 0, 1]])
    # i = 4
    # homgen_4_5 = np.array([[np.cos(dh_params[i,3]), -np.sin(dh_params[i,3]) * np.cos(dh_params[i,1]), np.sin(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.cos(dh_params[i,3])],
    #                     [np.sin(dh_params[i,3]), np.cos(dh_params[i,3]) * np.cos(dh_params[i,1]), -np.cos(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.sin(dh_params[i,3])],
    #                     [0, np.sin(dh_params[i,1]), np.cos(dh_params[i,1]), dh_params[i,2]],
    #                     [0, 0, 0, 1]])
    # i = 5
    # homgen_5_6 = np.array([[np.cos(dh_params[i,3]), -np.sin(dh_params[i,3]) * np.cos(dh_params[i,1]), np.sin(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.cos(dh_params[i,3])],
    #                     [np.sin(dh_params[i,3]), np.cos(dh_params[i,3]) * np.cos(dh_params[i,1]), -np.cos(dh_params[i,3]) * np.sin(dh_params[i,1]), dh_params[i,0] * np.sin(dh_params[i,3])],
    #                     [0, np.sin(dh_params[i,1]), np.cos(dh_params[i,1]), dh_params[i,2]],
    #                     [0, 0, 0, 1]])

    # H = (((((homgen_0_1 @ homgen_1_2) @ homgen_2_3) @ homgen_3_4) @ homgen_4_5) @ homgen_5_6)
    H = homgen_0_1.dot(homgen_1_2).dot(homgen_2_3).dot(homgen_3_4).dot(homgen_4_5)
    print(np.matrix(H))
    return H
    # H = np.dot(np.dot(np.dot(np.dot(np.dot(homgen_0_1, homgen_1_2)))))


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """

    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                            [0, np.sin(alpha), np.cos(alpha), d],
                            [0, 0, 0, 1]])


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    # R = np.array([[cos(phi), -sin(phi)][sin(phi), cos(phi)]])

    pass
