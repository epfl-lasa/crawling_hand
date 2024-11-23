#!/usr/bin/env python
#
# A parser for converting Python URDF objects into KDL Trees.
#
# Copyright (c) 2012, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Kelsey Hawkins

import numpy as np

import PyKDL as kdl
from urdf_parser_py.urdf import Robot


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
    cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
    return [sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy]


def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
                     kdl.Vector(*pos))


def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed':
        return kdl.Joint(jnt.name, kdl.Joint.Fixed)
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == 'revolute':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.TransAxis)
    print("Unknown joint type: %s." % jnt.joint_type)
    # return kdl.Joint(jnt.name, kdl.Joint.None)


def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(i.mass, origin.p,
                               kdl.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi


##
# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)

    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt,
                                      kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)

    add_children_to_tree(root)
    return tree


# def joint_list_to_kdl(q):
#     if q is None:
#         return None
#     if type(q) == np.matrix and q.shape[1] == 0:
#         q = q.T.tolist()[0]
#
#     q_kdl = kdl.JntArray(len(q))
#     for i, q_i in enumerate(q):
#         q_kdl[i] = q_i
#
#     return q_kdl

##### kdl conversions

def joint_to_kdl_jnt_array(q):
    if isinstance(q, np.ndarray) and q.ndim == 1:
        q_kdl = kdl.JntArray(q.size)
        for i in range(q.size):
            q_kdl[i] = q[i]

    elif isinstance(q, list):
        q_kdl = kdl.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i

    else:
        raise ValueError("Joint Vector q must be either a np.ndarray or list but is type {0}.".format(type(q)))

    return q_kdl


def kdl_jnt_array_to_joint(vec_kdl):
    assert isinstance(vec_kdl, kdl.JntArray)
    vec = np.zeros(vec_kdl.rows())

    for i in range(vec_kdl.rows()):
        vec[i] = vec_kdl[i]

    return vec


def kdl_vector_to_vector(vec_kdl):
    assert isinstance(vec_kdl, kdl.Vector), "Vector must be type 'kdl.Vector' but is '{0}'".format(type(vec_kdl))
    return np.array([vec_kdl.x(), vec_kdl.y(), vec_kdl.z()])


def vector_to_kdl_vector(x):
    assert (isinstance(x, np.ndarray) and x.size == 3) or (isinstance(x, list) and len(x) == 3)
    return kdl.Vector(x[0], x[1], x[2])


def kdl_inertia_to_matrix(inertia_kdl):
    inertia = np.zeros((3, 3))
    ix, iy = np.unravel_index(np.arange(9), (3, 3))
    for i in range(9):
        inertia[ix[i], iy[i]] = inertia_kdl[i]

    return inertia


def rot_x(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[1, 1] = + cos_theta
    mat[1, 2] = - sin_theta
    mat[2, 1] = + sin_theta
    mat[2, 2] = + cos_theta
    return mat


def rot_y(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[0, 0] = + cos_theta
    mat[0, 2] = + sin_theta
    mat[2, 0] = - sin_theta
    mat[2, 2] = + cos_theta
    return mat


def rot_z(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[0, 0] = + cos_theta
    mat[0, 1] = - sin_theta
    mat[1, 0] = + sin_theta
    mat[1, 1] = + cos_theta
    return mat


def kdl_matrix_to_mat(mat_kdl):
    mat = np.zeros((mat_kdl.rows(), mat_kdl.columns()))

    for i in range(mat_kdl.rows()):
        for j in range(mat_kdl.columns()):
            mat[i, j] = mat_kdl[i, j]

    return mat


def kdl_rot_to_mat(rot):
    return np.array([[rot[0, 0], rot[0, 1], rot[0, 2]],
                     [rot[1, 0], rot[1, 1], rot[1, 2]],
                     [rot[2, 0], rot[2, 1], rot[2, 2]]])


def kdl_frame_to_hom_transformation_matrix(frame):
    p = frame.p
    m = frame.M
    return np.array([[m[0, 0], m[0, 1], m[0, 2], p.x()],
                     [m[1, 0], m[1, 1], m[1, 2], p.y()],
                     [m[2, 0], m[2, 1], m[2, 2], p.z()],
                     [0, 0, 0, 1]])


def kdl_frame_to_transformation_matrix(frame):
    p = frame.p
    m = frame.M
    return np.array([[m[0, 0], m[0, 1], m[0, 2], p.x()],
                     [m[1, 0], m[1, 1], m[1, 2], p.y()],
                     [m[2, 0], m[2, 1], m[2, 2], p.z()]])


def rotation_mat_distance(mat_rot_1, mat_rot_2):
    mat_r = np.dot(mat_rot_1, mat_rot_2.transpose())
    theta = np.arccos((np.trace(mat_r) - 1.) / 2.)

    return theta


def main():
    import sys
    def usage():
        print("Tests for kdl_parser:\n")
        print("kdl_parser <urdf file>")
        print("\tLoad the URDF from file.")
        print("kdl_parser")
        print("\tLoad the URDF from the parameter server.")
        sys.exit(1)

    if len(sys.argv) > 2:
        usage()
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
    if (len(sys.argv) == 1):
        robot = Robot.from_parameter_server()
    else:
        f = file(sys.argv[1], 'r')
        robot = Robot.from_xml_string(f.read())
        f.close()
    tree = kdl_tree_from_urdf_model(robot)
    num_non_fixed_joints = 0
    for j in robot.joint_map:
        if robot.joint_map[j].joint_type != 'fixed':
            num_non_fixed_joints += 1
    print("URDF non-fixed joints: %d;" % num_non_fixed_joints)
    print("KDL joints: %d" % tree.getNrOfJoints())
    print("URDF joints: %d; KDL segments: %d" % (len(robot.joint_map),
                                                 tree.getNrOfSegments()))
    import random
    base_link = robot.get_root()
    end_link = robot.link_map.keys()[random.randint(0, len(robot.link_map) - 1)]
    chain = tree.getChain(base_link, end_link)
    print("Root link: %s; Random end link: %s" % (base_link, end_link))
    for i in range(chain.getNrOfSegments()):
        print(chain.getSegment(i).getName())


if __name__ == "__main__":
    main()
