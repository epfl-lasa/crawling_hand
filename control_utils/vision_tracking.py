import rospy

import time
import numpy as np
import tools.rotations as rot

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Int32

class objects_tracking:
    """
    get the poses of the crawling robot (a small marker on the palm)
    get the poses of objects to be grasped, using the icg_ros package
    """

    def __init__(self, icg_objects_names=['yellow', 'blue', 'red'], nums=360):
        rospy.init_node('objects_tracker', anonymous=True, )

        self.icg_objects_names = icg_objects_names
        self._icg_objects_poses = np.zeros([len(icg_objects_names), 7])
        self._icg_objects_poses_optical = np.zeros([len(icg_objects_names), 7])

        # rospy.Subscriber("objects_pose", PoseArray, self.icg_pose_cb, queue_size=10)
        rospy.Subscriber("object_poses", PoseArray, self.icg_pose_cb, queue_size=10)
        rospy.Subscriber("aruco_simple/pose", Pose, self.aruco_marker_1_pose_cb, queue_size=10)
        rospy.Subscriber("aruco_simple/pose2", Pose, self.aruco_marker_2_pose_cb, queue_size=10)
        rospy.Subscriber("aruco_single_test/pose", PoseStamped, self.aruco_marker_3_pose_cb, queue_size=10)
        rospy.Subscriber("keyboard_cmd", Int32, self.keyboard_cb, queue_size=10)

        self._key = 1

        self._m1 = None  # the first marker, on the table as origin
        self.m1_size = 0.067
        self.m1_updated = False

        self._m2 = None  # the second marker, on the palm
        self.m2_size = 0.03
        self._m3 = None   # the 3rd marker, on the iiwa end-effector, exact size
        time.sleep(0.2)

        self.m2_in_palm = np.array([0.06, 0, 0, 1, 0, 0, 0])
        self.m2_in_palm[3:] = rot.euler2quat([0, 0, np.pi / 2])
        # rospy.spin
        self.nums = nums
        q_proj = [rot.euler2quat([0, 0, i / 180 * np.pi]) for i in range(nums)]
        self.q_proj = np.vstack(q_proj)

        self.min_quat_dis = np.zeros(len(icg_objects_names))

    def keyboard_cb(self, state: Int32):
        self._key = state.data

    def icg_pose_cb(self, state: PoseArray):
        if len(state.poses) == len(self.icg_objects_names):
            for i in range(len(self.icg_objects_names)):
                # obj pose in the camera_color_optical_frame
                pose = np.array([state.poses[i].position.x, state.poses[i].position.y, state.poses[i].position.z,
                                 state.poses[i].orientation.w, state.poses[i].orientation.x, state.poses[i].orientation.y,
                                 state.poses[i].orientation.z])
                self._icg_objects_poses_optical[i, :] = pose

    def aruco_marker_1_pose_cb(self, state: Pose):
        if not self.m1_updated:
            pose = np.array(
                [state.position.x, state.position.y, state.position.z, state.orientation.w, state.orientation.x,
                 state.orientation.y, state.orientation.z])

            self._m1 = pose
            self.m1_updated = True

    def aruco_marker_2_pose_cb(self, state: Pose):
        pose = np.array([state.position.x, state.position.y, state.position.z, state.orientation.w, state.orientation.x,
                         state.orientation.y, state.orientation.z])
        pose[:3] = pose[:3] / self.m1_size * self.m2_size
        self._m2 = pose

    def aruco_marker_3_pose_cb(self, state: PoseStamped):
        pose = np.array([state.pose.position.x, state.pose.position.y, state.pose.position.z, state.pose.orientation.w, state.pose.orientation.x,
                         state.pose.orientation.y, state.pose.orientation.z])
        self._m3 = pose

    @property
    def m3(self):
        """
        pose of the palm in the 1st marker frame
        :return:
        """

        if self.m1_updated:
            T = np.linalg.inv(rot.pose2T(self.m1)) @ rot.pose2T(self._m3)  # m3 in m1
            return rot.T2pose(T)

        else:
            print('Marker 1 is not detected.')
            return None

    @property
    def m2(self):
        """
        pose of the center of the palm in the 1st marker frame (table)
        :return:
        """

        if self.m1_updated:
            T = np.linalg.inv(rot.pose2T(self.m1)) @ rot.pose2T(self._m2)  # m2 in m1
            T = T @ np.linalg.inv(rot.pose2T(self.m2_in_palm))  # palm in m1
            return rot.T2pose(T)

        else:
            print('Marker 1 is not detected.')
            return None
            # return np.array([0.05512771,  0.27072045,  0.11335479, 1,0,0,0])

    @property
    def m1(self):
        """
        pose of the 1st marker in the color_optical_frame
        1st marker is on teh table
        :return:
        """
        return self._m1
        # return np.array([0.40748814, -0.17603464,  0.67684108 , 1,0,0,0])

    @property
    def x_obj(self) -> list:
        """
        poses of objects in the 1st marker frame (id 18)
        :return:
        """
        icg_objects_poses = []
        if self.m1_updated:
            for i in range(len(self.icg_objects_names)):
                # obj pose in the marker 1 frame
                T = np.linalg.inv(rot.pose2T(self.m1)) @ rot.pose2T(self._icg_objects_poses_optical[i, :])
                tmp = rot.T2pose(T)
                dis_all = rot.ori_dis(np.repeat(tmp[3:].reshape(1, -1), self.nums, axis=0), self.q_proj)
                q_nearest = self.q_proj[np.argmin(dis_all), :]
                pose = np.concatenate([tmp[:3], q_nearest])
                icg_objects_poses.append(pose)
                self.min_quat_dis[i] = np.min(dis_all)
            return icg_objects_poses

    @property
    def key(self):
        return self._key

if __name__ == '__main__':
    v = objects_tracking()
    print(v.m1)
    print(v.m2)
    print(v.x_obj)
