import copy

import numpy as np
import rospy
import actionlib

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal,FollowJointTrajectoryGoal  # it's passed to the action server itself at initialization.
from control_msgs.msg import \
    FollowJointTrajectoryFeedback  # it's published without using the action server (topic /state)
from control_msgs.msg import FollowJointTrajectoryResult  # it's only passaed to the action server.
# curiously, it's always passed as an error (set_aborted method)

from control_msgs.msg import FollowJointTrajectoryActionFeedback

# from builtin_interfaces.msg import Duration
# from action_msgs.msg import GoalStatus
from std_msgs.msg import Float64MultiArray
import time


class real_robot:
    def __init__(self, control_mode='position', node_name='crawling_robot', print_cmd=False, legs=5):
        self._get_result_future = None
        rospy.init_node('real_hand_ros1', anonymous=True, )
        rospy.Subscriber("joint_states", JointState, self.listener_callback, queue_size=10)

        # self.joint_names = ['joint' + str(i) for i in range(15)]
        if legs == 4:
            self.joint_names = ['joint' + str(i) for i in range(0, 6)] + ['joint' + str(i) for i in range(9, 15)]
        else:
            self.joint_names = ['joint' + str(i) for i in range(0, 20)]

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
        self.joint_limits = np.array(
            [[-1, -1.4, -1.4, -1.4] * 5,
             [1, 1.4, 1.4, 1.4] * 5])

        self.feedback = FollowJointTrajectoryFeedback()
        self.client = actionlib.SimpleActionClient('joint_trajectory_controller/follow_joint_trajectory',
                                                   FollowJointTrajectoryAction)
        self.client.wait_for_server()

        # # action client
        # self._action_client = ActionClient(self, FollowJointTrajectory,
        #                                    '/joint_trajectory_controller/follow_joint_trajectory')
        self.__send_goal_future = None
        self.__remaining_iteration = 0
        self.dt = 0.002

    # # def pos_mid_callback(self, pos: Float64MultiArray):
    #     positions = pos.data
    #     self.pos_mid_cmd = np.array(positions)
    #     if self.print_cmd:
    #         print(time.time() - self.t0, np.array(positions))

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

    def move_to_joints(self, q: np.ndarray, t=2, sim2real=False):
        """
            position control mode
        :param sim2real:
        :param q: (n, 20), n>=1
        :return:
        """
        q = np.clip(q, self.joint_limits[0, :], self.joint_limits[1, :])
        goal_message = FollowJointTrajectoryGoal()
        goal_message.trajectory = JointTrajectory()
        goal_message.trajectory.joint_names = self.joint_names
        for i in range(q.shape[0]):
            # print(i)
            # q_i = last_two_joints_collision(q[i, :])
            q_i = q[i, :]
            t_all = t / q.shape[0] * (i + 1)
            ts = int(t_all)
            tn = int((t_all - ts) * 1e9)
            trajectory_point = JointTrajectoryPoint(positions=list(q_i),
                                                    time_from_start=rospy.Duration(secs=ts,
                                                                             nsecs=tn))
            goal_message.trajectory.points.append(trajectory_point)
        #
        self.client.send_goal_and_wait(goal_message)
        # return self.client.wait_for_result()
        # future = self._action_client.send_goal_async(goal_message)
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
    time.sleep(0.4)

    print(robot.q)

    q0 = np.copy(robot.q)
    # q0 = np.zeros(robot.legs * 3) + 0.2
    robot.move_to_joints(q0, t=2)
    #
    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)




if __name__ == '__main__':
    main()
