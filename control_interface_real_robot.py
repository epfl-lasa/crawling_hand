import copy

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from std_msgs.msg import Float64MultiArray
import time


def q_sim2real(q: np.ndarray, legs=4):
    """
    transfer the joints from simulation to real robot
    :param q: (20, ) or (n, 20), joint position from MuJoCo
    :return: (n, 12) joint cmd for the real robot
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
    assert q.shape[1] == 20
    if legs == 4:
        q_cmd = np.concatenate([q[:, :3], q[:, 4:7], q[:, 12:15], q[:, 16:19]], axis=1)  # missing one leg from Kai
        q_cmd[:, [0, 3, 6, 9]] = - q_cmd[:, [0, 3, 6, 9]]  # reverse the first joint
    elif legs == 5:
        q_cmd = np.concatenate([q[:, :3], q[:, 4:7], q[:, 8:11], q[:, 12:15], q[:, 16:19]],
                               axis=1)  # missing one leg from Kai
        q_cmd[:, [0, 3, 6, 9, 12]] = - q_cmd[:, [0, 3, 6, 9, 12]]  # reverse the first joint
    else:
        raise NotImplementedError

    return q_cmd


def last_two_joints_collision(q_i):
    """
    avoid collision for last two fingers
    :param q_i: (12, )
    :return:
    """
    if q_i[6] - q_i[9] > 1:  # the first joint of last two fingers
        # self.update_data()
        delta_q = (q_i[6] - q_i[9] - 0.7) / 2
        q_i[6] -= delta_q
        q_i[9] += delta_q
        print('Possible collision between last two fingers, modify joints')
    return q_i


# class vel_mid(Node):
#     def __init__(self, num=12):
#         self.num = num
#         self.zero_vel = np.zeros(num)
#         rclpy.init()
#         self.node = super().__init__('vel_mid')
#         self.vel_publisher_ = self.create_publisher(Float64MultiArray,
#                                                         '/velocity_controller/commands', 10)
#         self.pos_mid_sub_ = self.create_subscription(Float64MultiArray, '/velocity_mid_controller/commands',
#                                                      self.pos_mid_callback, 10)
#         self.pos_mid_cmd = np.zeros(num)
#
#     def pos_mid_callback(self, pos: Float64MultiArray):
#         positions = pos.data
#         self.pos_mid_cmd = np.array(positions)
#
#     def velocity_control(self):
#         """
#         publish joint velocity command
#         :return:
#         """
#         vel_msg = Float64MultiArray()
#         vel_msg.data = list(self.vel_mid_cmd)
#         self.vel_publisher_.publish(vel_msg)


class real_robot(Node):
    def __init__(self, control_mode='position', node_name='crawling_robot', print_cmd=False, legs=4):
        self._get_result_future = None
        rclpy.init()
        self.node = super().__init__(node_name)
        self.subscription = self.create_subscription(JointState, 'joint_states', self.listener_callback, 10)
        # self.joint_names = ['joint' + str(i) for i in range(15)]
        if legs == 4:
            self.joint_names = ['joint' + str(i) for i in range(0, 6)] + ['joint' + str(i) for i in range(9, 15)]
        else:
            self.joint_names = ['joint' + str(i) for i in range(0, 15)]

        print("leg num:", legs)
        self.legs = legs
        self.q_dict = {i: 0 for i in self.joint_names}
        self.dq_dict = {i: 0 for i in self.joint_names}
        self.ddq_dict = {i: 0 for i in self.joint_names}
        self.num = len(self.joint_names)
        self._q = np.zeros(self.num)
        self._dq = np.zeros(self.num)
        self._effort = np.zeros(self.num)
        assert control_mode in ['position', 'velocity']
        self.control_mode = control_mode
        self.print_cmd = print_cmd
        self.t0 = time.time()

        # joint limits
        self.leg_nums = int(self.num / 3)
        self.joint_limits = np.array(
            [[-1, -1.5, -1.4, -1, -1.5, -1.4, -0.8, -1.5, -1.4, -0.7, -1.4, -1.4, -1, -1.4, -1.4],
             [1, 1.5, 1.4, 1, 1.5, 1.4, 0.8, 1.5, 1.4, 0.7, 1.4, 1.4, 1, 1.4, 1.4]])

        #  ros2 control switch_controllers --activate joint_state_broadcaster --deactivate joint_trajectory_controller --activate velocity_controller
        self.vel_publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.pos_publisher_ = self.create_publisher(Float64MultiArray, '/velocity_mid_controller/commands', 10)
        self.pos_mid_sub_ = self.create_subscription(Float64MultiArray, '/velocity_mid_controller/commands',
                                                     self.pos_mid_callback, 10)
        self.pos_mid_cmd = np.zeros(self.num)

        # self.cmd_ = FollowJointTrajectory().Goal()
        # # self
        # self.cmd_.trajectory.joint_names = self.joint_names
        # self.cmd_.joint_names = self.joint_names
        # self.cmd_.points.

        # action client
        self._action_client = ActionClient(self, FollowJointTrajectory,
                                           '/joint_trajectory_controller/follow_joint_trajectory')
        self.__send_goal_future = None
        self.__remaining_iteration = 0
        self.dt = 0.002

    def pos_mid_callback(self, pos: Float64MultiArray):
        positions = pos.data
        self.pos_mid_cmd = np.array(positions)
        if self.print_cmd:
            print(time.time() - self.t0, np.array(positions))

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        return result

    def listener_callback(self, msg: JointState):
        q_name = list(msg.name)
        for i in range(len(q_name)):
            self.q_dict[q_name[i]] = msg.position[i]
            self.dq_dict[q_name[i]] = msg.velocity[i]
            self.ddq_dict[q_name[i]] = msg.effort[i]
        self._q = np.array(list(self.q_dict.values()))
        self._dq = np.array(list(self.dq_dict.values()))
        self._effort = np.array(list(self.ddq_dict.values()))
        # print(self._q)

    def update_data(self):
        """
        update data from all callback
        :return:
        """
        rclpy.spin_once(self)  # it takes 16 ms

    def vel_based_pos_control(self, q: np.ndarray, vel_norm=0.4):
        """
        move to desired q by using a DS, it has a continuous vel.
        Need to have a while loop outside of this function to keep sending
        :param q:
        :param vel_norm:
        :return:
        """
        q = q.flatten()
        assert len(q) == self.legs * 3
        q = np.clip(q, self.joint_limits[0, :], self.joint_limits[1, :])
        q = last_two_joints_collision(q)

        # self.update_data()
        error = np.linalg.norm((self.q - q))
        if error > 0.01:
            desired_vel = - vel_norm * (self.q - q) / (error + 0.5)
            if self.print_cmd:
                print(time.time() - self.t0, error)
        else:
            desired_vel = np.zeros(self.num)

        self.__velocity_control(desired_vel)

    def send_position(self, pos: np.ndarray):
        """
        publish joint position command
        :return:
        """
        q = pos.flatten()
        assert len(q) == self.legs * 3
        pos_msg = Float64MultiArray()
        pos_msg.data = list(q)
        self.pos_publisher_.publish(pos_msg)

    def __velocity_control(self, vel: np.ndarray):
        """
        publish joint velocity command
        :return:
        """
        vel_msg = Float64MultiArray()
        vel_msg.data = list(vel)
        self.vel_publisher_.publish(vel_msg)

    def move_to_joints_by_vel(self, q: np.ndarray, vel_norm=0.4):
        """
        move to desired q by velocity control, it will stop at q,

        :param q: (12, ) desired q,
        :param vel_norm: float, norm of vel
        :return:
        """
        q = q.flatten()
        assert len(q) == self.legs * 3
        q = np.clip(q, self.joint_limits[0, :], self.joint_limits[1, :])
        q = last_two_joints_collision(q)

        error = 1
        while error > 0.01:
            self.update_data()
            error = np.linalg.norm((self.q - q))
            desired_vel = - vel_norm * (self.q - q) / (error + 0.001)
            self.__velocity_control(desired_vel)
            print('error', error)
        print('Done,', error, np.linalg.norm((self.q - q)))
        self.__velocity_control(np.zeros(12))  # motors will stop with zero velocities

    def move_to_joints(self, q: np.ndarray, t=2, sim2real=False):
        """
            position control mode
        :param sim2real:
        :param q: (n, 20), n>=1
        :return:
        """
        if sim2real:
            q = q_sim2real(q, legs=self.legs)  # (n , 12)

        q = np.clip(q, self.joint_limits[0, :], self.joint_limits[1, :])
        goal_message = FollowJointTrajectory.Goal()
        goal_message.trajectory.joint_names = self.joint_names
        for i in range(q.shape[0]):
            # print(i)
            # q_i = last_two_joints_collision(q[i, :])
            q_i = q[i, :]
            t_all = t / q.shape[0] * (i + 1)
            ts = int(t_all)
            tn = int((t_all - ts) * 1e9)
            trajectory_point = JointTrajectoryPoint(positions=list(q_i),
                                                    time_from_start=Duration(sec=ts,
                                                                             nanosec=tn))
            goal_message.trajectory.points.append(trajectory_point)
        #
        future = self._action_client.send_goal_async(goal_message)
        # rclpy.spin_until_future_complete(self, future)
        # print(self.q - q)

    @property
    def q(self):
        return self._q

    @property
    def dq(self):
        return self._dq

    @property
    def effort(self):
        return self._effort


def main(args=None):
    # rclpy.init(args=args)

    ############ position control test
    robot = real_robot()

    robot.update_data()
    print(robot.q)

    q0 = np.copy(robot.q)
    q0 = np.zeros(robot.legs * 3) + 0.2
    robot.move_to_joints(q0, t=2)
    #
    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    robot.destroy_node()
    rclpy.shutdown()


def velocity_control():
    #################  vel intermediate node
    robot = real_robot(node_name='vel_mid', print_cmd=True, legs=5)
    while np.linalg.norm(robot.q) < 1e-10:  # make sure that we can receive position data from motors
        rclpy.spin_once(robot)
        time.sleep(0.01)

    robot.pos_mid_cmd = copy.deepcopy(robot.q)
    print("ready, start velocity control")
    while 1:
        rclpy.spin_once(robot)  # update pos_mid_cmd
        robot.vel_based_pos_control(robot.pos_mid_cmd, vel_norm=3)
        time.sleep(0.002)


if __name__ == '__main__':
    # main()
    velocity_control()
