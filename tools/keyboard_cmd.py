#!/usr/bin/env python
# receive the keyboard input as number

#              right left
#gripper-open   10    20
#gripper-close  11    21 
#
#ball-position
#change x->y    1xy   2xy


import rospy
from std_msgs.msg import Int32

pub = rospy.Publisher('keyboard_cmd', Int32, queue_size=1)
rospy.init_node('keyboard_cmd_node', anonymous=True)

print ('''Input state: 
                          right  left
        gripper-open      10     20
        gripper-close     11     21
        gripper-decrease  1
        Exit while        0      
        Exit!!!           9''')
#print ('Input state: ')
a = int(input('input cmd: '))

while a!=9:
    if a == 10 or a == 11 or a == 20 or a ==21:
        msg = Int32()
        msg.data = a
        pub.publish(msg)
        print ('gripper cmd: ',a)
    elif a == 1:
        msg = Int32()
        msg.data = a
        pub.publish(msg)
        print ('decrease the gripper ',a)
    elif a ==0:
        msg = Int32()
        msg.data = a
        pub.publish(msg)
        print ('Exit while loop ',a) 
    else:
        rospy.WARN('Wrong number input!')
    print ('''Input state: 
                          right  left
        gripper-open      10     20
        gripper-close     11     21
        gripper-decrease  1
        Exit while        0      
        Exit!!!           9''')

    a = int(input('input cmd: '))
            





