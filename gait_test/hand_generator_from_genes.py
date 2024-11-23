import numpy as np

# import xml.etree.ElementTree as ET
import tools.rotations as rot

class crawling_hand:
    def __init__(self, fingers, dof, link_lengths, objects=True, d_angle=np.pi/4, real=False):
        """

        :param fingers: [1,1,1,0...], (12,), 12 points is located on the circle of the palm
        :param dof:  (12,), for each of it, dof = 2,3 or 4
        :param link_lengths: (12, ) fingers might have different lengths, assuming the lengths of links are the same for
          one finger
        :param objects: adding objects into the xml or not
        """
        assert len(fingers) == 8
        assert len(dof) == 8
        #
        self.fingers = fingers
        self.finger_nums = sum(fingers)
        self.dof = dof
        self.link_lengths = link_lengths
        #  build the palm as a cylinder
        self.palm_radius = 0.05
        self.palm_height = 0.01
        self.capsule_size = 0.01
        self.d_angle = d_angle

        self.objects = objects
        self.real=real # if real robot size
        if self.real:
            assert len(link_lengths) == 5 # lengths for a single finger
            self.palm_radius = link_lengths[0]
            self.capsule_size = 0.015
        else:
            assert len(link_lengths) == 8

    def return_xml_ycb_real(self, q_grasp=None, q_used=None, ycb_poses=None, color=None,used_objs=None, version=1):
        # self.palm_height = 0.03
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix_real()
        ii = -1
        ii_max = int(sum(q_used)/4)
        removed_joints = []
        for i in range(8):  # for each position
            if self.fingers[i]:  # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]  # 4 in default
                x = self.palm_radius * np.cos(theta + np.pi/2)  # position of the finger start point
                y = self.palm_radius * np.sin(theta + np.pi/2)
                quat = rot.euler2quat([0, 0, theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])


                # always the first two fingers used
                if i not in [0, 1]:
                    finger_xml = '''
                                    <body name='finger_''' + str(i) + ''' ' pos=' ''' + str(x) + ' ' + str(y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                <body name='MCP_spread_motor_''' + str(i) + '''' pos='0 0 0.0036'>
                                    <inertial pos='0.000245265 -0.00496883 -0.0114693' quat='0.524663 0.474196 -0.523113 0.475625' mass='0.21816' diaginertia='5.65844e-05 4.36866e-05 3.19042e-05' />
                                    <joint name='joint_''' + str(i) + str(0) + '''' type='hinge' pos='0 0 0' axis='0 0 -1' range='-2 2' />
                                    <geom pos='0 0 -0.0036' type='mesh' rgba='0.7 0.7 0.7 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                    <geom pos='0 0 -0.0036' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                    <body name='MCP_motor_''' + str(i) + '''' pos='0.0106 0.0235 -0.0116'>
                                        <inertial pos='-0.0103512 0.0125458 3.30732e-06' quat='0.535591 0.461692 -0.535523 0.461734' mass='0.151474' diaginertia='3.89252e-05 3.18099e-05 1.61153e-05' />
                                        <joint name='joint_''' + str(i) + str(1) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                        <geom pos='-0.0106 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                        <geom pos='-0.0106 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                        <body name='PIP_DIP_motor_''' + str(i) + '''' pos='0 0.039 0'>
                                            <inertial pos='-0.010709 0.0103943 3.56469e-06' quat='0.533676 0.463904 -0.533611 0.463942' mass='0.140537' diaginertia='2.81223e-05 2.20307e-05 1.40916e-05' />
                                            <joint name='joint_''' + str(i) + str(2) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                            <geom pos='-0.0106 -0.0625 0.008' type='mesh' rgba='0 0.7 0 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                            <geom pos='-0.0106 -0.0625 0.008' type='mesh' rgba='0 0.7 0 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                            <body name='distal_''' + str(i) + ''' ' pos='0.0029 0.0305 0'>
                                                <inertial pos='-0.0135 0.0167496 0' quat='0.5 0.5 -0.5 0.5' mass='0.112487' diaginertia='1.8e-05 1.7e-05 8e-06' />
                                                <joint name='joint_''' + str(i) + str(3) + '''' pos='0 0 0' axis='-1 0 0'  range='-2 2'/>
                                                <geom pos='-0.0135 -0.093 0.008' type='mesh' rgba='0 0.7 0 1' mesh='distal_1' name='distal_''' + str(
                        i) + ''''/>
                                                <site name='finger_''' + str(i) + '''_tip' pos='-0.0135 0.026 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                         </body>
                                    '''
                else:
                    quat_str_i_j_all = []
                    for j, axis in enumerate([np.array([0, 0, -1]), np.array([-1, 0, 0]), np.array([-1, 0, 0]), np.array([-1, 0, 0])]):
                        axis_angle = axis * (q_grasp[i * 4 + j])
                        quat_i_j = rot.axisangle2quat(axis_angle)
                        quat_str_i_j = str(quat_i_j[0]) + ' ' + str(quat_i_j[1]) + " " + str(quat_i_j[2]) + " " + str(quat_i_j[3])
                        quat_str_i_j_all.append(quat_str_i_j)
                    finger_xml = '''
                                                        <body name='finger_''' + str(i) + '''' pos=' ''' + str(
                        x) + ' ' + str(y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                                    <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                                    <body name='MCP_spread_motor_''' + str(i) + '''' pos='0 0 0.0036' quat=' ''' +quat_str_i_j_all[0]  + ''' '   >
                                                        <inertial pos='0.000245265 -0.00496883 -0.0114693' quat='0.524663 0.474196 -0.523113 0.475625' mass='0.21816' diaginertia='5.65844e-05 4.36866e-05 3.19042e-05' />
                                                        <geom pos='0 0 -0.0036' type='mesh' rgba='0.7 0.7 0.7 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                                        <geom pos='0 0 -0.0036' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                                        <body name='MCP_motor_''' + str(i) + '''' pos='0.0106 0.0235 -0.0116' quat=' ''' +quat_str_i_j_all[1]  + ''' '   >
                                                            <inertial pos='-0.0103512 0.0125458 3.30732e-06' quat='0.535591 0.461692 -0.535523 0.461734' mass='0.151474' diaginertia='3.89252e-05 3.18099e-05 1.61153e-05' />
                                                            <geom pos='-0.0106 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                                            <geom pos='-0.0106 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                                            <body name='PIP_DIP_motor_''' + str(i) + '''' pos='0 0.039 0' quat=' ''' +quat_str_i_j_all[2]  + ''' '   >
                                                                <inertial pos='-0.010709 0.0103943 3.56469e-06' quat='0.533676 0.463904 -0.533611 0.463942' mass='0.140537' diaginertia='2.81223e-05 2.20307e-05 1.40916e-05' />
                                                                <geom pos='-0.0106 -0.0625 0.008' type='mesh' rgba='0 0.7 0 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                                                <geom pos='-0.0106 -0.0625 0.008' type='mesh' rgba='0 0.7 0 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                                                <body name='distal_''' + str(i) + '''' pos='0.0029 0.0305 0' quat=' ''' +quat_str_i_j_all[3]  + ''' '   >
                                                                    <inertial pos='-0.0135 0.0167496 0' quat='0.5 0.5 -0.5 0.5' mass='0.112487' diaginertia='1.8e-05 1.7e-05 8e-06' />
                                                                    <geom pos='-0.0135 -0.093 0.008' type='mesh' rgba='0 0.7 0 1' mesh='distal_1' name='distal_''' + str(
                        i) + ''''/>
                                                                    <site name='finger_''' + str(i) + '''_tip' pos='-0.0135 0.026 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                                                </body>
                                                            </body>
                                                        </body>
                                                    </body>
                                             </body>
                                                        '''
                xml_data += finger_xml



        if ycb_poses.shape == (7,):
            ycb_poses = ycb_poses.reshape(1,-1)
        if not self.objects:
            if ycb_poses is not None:
                xml_data += '''</body>
                <include file="../descriptions/objs/ycb_bodies.xml"/>
                </worldbody>

                <actuator>
                '''
            else:
                xml_data += '''
                </worldbody>

                <actuator>
                '''
        else:  # add objects into xml
            if q_grasp is not None:
                pass
                # add spheres which are fixed in the hand base frame
                obj_nums = 3
                rgba_all = []
                for index, obj in enumerate(list(used_objs)): # e.g. used_objs =[1,0,2]
                    obj_name = self.objects[obj]
                    xyz = ycb_poses[index,:]
                    xyz_str = str(xyz[0]) + " " + str( xyz[1]) + " " + str(xyz[2])
                    quat_str = str(xyz[3]) + " " + str( xyz[4]) + " " + str(xyz[5])  + " " + str(xyz[6])
                    if obj ==0:
                        obj_xml = '''
                        <body name=\"sphere_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
        <geom name=\"sphere_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.017\" type=\"sphere\"/>
   </body>          
   '''
                    elif obj==1:
                        obj_xml = '''
                                 <body name=\"cylinder_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
        <geom name=\"cylinder_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.012 0.045\" type=\"cylinder\"/>
   </body> 
                           '''
                    elif obj == 2:
                        obj_xml = '''
                        <body name=\"box_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
        <geom name=\"box_1\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
    </body>
                        '''
                    elif obj == 3:
                        obj_xml = '''
                                           <body name=\"box_2\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
        <geom name=\"box_2\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 1 0.2 1\" size=\"0.015 0.0075 0.03\" type=\"box\"/>
    </body>
                                           '''
                    elif obj ==4:
                        obj_xml = '''
                                                                  <body name=\"box_3\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                               <geom name=\"box_3\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
                           </body>
                                                                  '''
                    else:
                        raise NotImplementedError

                    xml_data += obj_xml

                xml_data += '''
                        </body>
                        </body>
                        '''  # finish hand
                xml_data += '''
                        </worldbody>

                            <actuator>
                '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>

                        <include file="../descriptions/objs/box.xml"/>

                            </worldbody>

                            <actuator>
                            '''

        #### actuators:
        for i in range(8):
            if self.fingers[i] and i > 1:  # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    xml_data += '''<motor name=\"joint_''' + str(i) + str(
                        j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                '''
        xml_data += '''</actuator>

        </mujoco>
        '''

        return xml_data

    def return_xml_ycb_real_v2(self, q_grasp=None, q_used=None, ycb_poses=None, color=None, used_objs=None, version=1):
        # self.palm_height = 0.03
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix_real_v2()
        ii = -1
        ii_max = int(sum(q_used) / 4)
        removed_joints = []
        for i in range(8):  # for each position
            if self.fingers[i]:  # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]  # 4 in default
                x = self.palm_radius * np.cos(theta + np.pi / 2)  # position of the finger start point
                y = self.palm_radius * np.sin(theta + np.pi / 2)
                quat = rot.euler2quat([0, 0, theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])

                # always the first two fingers used
                if i not in [0, 1]:
                    finger_xml = '''
                                     <body name='finger_''' + str(i) + ''' ' pos=' ''' + str(x) + ' ' + str(
                        y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                 <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                 <body name='MCP_spread_motor_''' + str(i) + '''' pos='0 0 0.0036'>
                                     <inertial pos='0.00780713 0.0116002 -0.00421738' quat='0.556955 0.285031 -0.717518 0.306149' mass='0.218111' diaginertia='7.16804e-05 5.53793e-05 3.8057e-05'/>
                                    <joint name='joint_''' + str(i) + str(0) + '''' type='hinge' pos='0 0 0' axis='0 0 -1' range='-2 2' />
                                     <geom type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                     <geom pos='0 0.009 0' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                     <body name='MCP_motor_''' + str(i) + '''' pos='-0.0135 0.0325 -0.008'>
                                          <inertial pos='0.0133661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                          <joint name='joint_''' + str(i) + str(1) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                         <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                         <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                         <body name='PIP_DIP_motor_''' + str(i) + '''' pos='0 0.049 0'>
                                             <inertial pos='0.0123661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                            <joint name='joint_''' + str(i) + str(2) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                             <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                             <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='.7 0.7 0.7 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                             <body name='distal_''' + str(i) + ''' ' pos='0 0.049 0'>
                                                 <inertial pos='0.0118444 0.0123626 3.07129e-06' quat='0.503638 0.496337 -0.503565 0.496408' mass='0.163525' diaginertia='3.3711e-05 2.87101e-05 1.70008e-05'/>
                                                 <joint name='joint_''' + str(i) + str(3) + '''' pos='0 0 0' axis='-1 0 0'  range='-2 2'/>
                                                 <geom pos='0.0125 -0.1215 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='DIP_motor_1' name='distal_''' + str(
                        i) + ''''/>
                                                <geom class='sliding_finger' pos='0.0125 -0.1215 0.008' rgba='0 0.7 0 1' type='mesh' mesh='distal_1' name='distal_tip_''' + str(
                        i) + ''''/>
                                                 <site name='finger_''' + str(i) + '''_tip' pos='0.012 0.03 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                             </body>
                                         </body>
                                     </body>
                                 </body>
                          </body>
                                     '''
                else:
                    quat_str_i_j_all = []
                    for j, axis in enumerate(
                            [np.array([0, 0, -1]), np.array([-1, 0, 0]), np.array([-1, 0, 0]), np.array([-1, 0, 0])]):
                        axis_angle = axis * (q_grasp[i * 4 + j])
                        quat_i_j = rot.axisangle2quat(axis_angle)
                        quat_str_i_j = str(quat_i_j[0]) + ' ' + str(quat_i_j[1]) + " " + str(quat_i_j[2]) + " " + str(
                            quat_i_j[3])
                        quat_str_i_j_all.append(quat_str_i_j)
                    finger_xml = '''
                                                         <body name='finger_''' + str(i) + '''' pos=' ''' + str(
                        x) + ' ' + str(y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                                     <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                                     <body name='MCP_spread_motor_''' + str(
                        i) + ''''  quat=' ''' + quat_str_i_j_all[0] + ''' '   >
                                                          <inertial pos='0.00780713 0.0116002 -0.00421738' quat='0.556955 0.285031 -0.717518 0.306149' mass='0.218111' diaginertia='7.16804e-05 5.53793e-05 3.8057e-05'/>
                                                         <geom type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                                         <geom pos='0 0.009 0' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                                         <body name='MCP_motor_''' + str(
                        i) + '''' pos='-0.0135 0.0325 -0.008' quat=' ''' + quat_str_i_j_all[1] + ''' '   >
                                                             <inertial pos='0.0133661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                                             <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                                             <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                                             <body name='PIP_DIP_motor_''' + str(
                        i) + '''' pos='0.001 0.049 0' quat=' ''' + quat_str_i_j_all[2] + ''' '   >
                                                                 <inertial pos='0.0118444 0.0123626 3.07129e-06' quat='0.503638 0.496337 -0.503565 0.496408' mass='0.163525' diaginertia='3.3711e-05 2.87101e-05 1.70008e-05'/>
                                                                 <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                                                 <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='.7 0.7 0.7 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                                                 <body name='distal_''' + str(
                        i) + '''' pos='0 0.049 0' quat=' ''' + quat_str_i_j_all[3] + ''' '   >
                                                                     <inertial pos='-0.0135 0.0167496 0' quat='0.5 0.5 -0.5 0.5' mass='0.112487' diaginertia='1.8e-05 1.7e-05 8e-06' />
                                                                     <geom pos='0.0125 -0.1215 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='DIP_motor_1' name='distal_''' + str(
                        i) + ''''/>                                 
                                                                    <geom class='sliding_finger' pos='0.0125 -0.1215 0.008' rgba='0 0.7 0 1' type='mesh' mesh='distal_1' name='distal_tip_''' + str(
                        i) + ''''/>
                                                                     <site name='finger_''' + str(i) + '''_tip' pos='0.012 0.03 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                                                 </body>
                                                             </body>
                                                         </body>
                                                     </body>
                                              </body>
                                                         '''
                xml_data += finger_xml

        if ycb_poses.shape == (7,):
            ycb_poses = ycb_poses.reshape(1, -1)
        if not self.objects:
            if ycb_poses is not None:
                xml_data += '''</body>
                 <include file="../descriptions/objs/ycb_bodies.xml"/>
                 </worldbody>

                 <actuator>
                 '''
            else:
                xml_data += '''
                 </worldbody>

                 <actuator>
                 '''
        else:  # add objects into xml
            if q_grasp is not None:
                # add spheres which are fixed in the hand base frame
                obj_nums = 3
                rgba_all = []
                for index, obj in enumerate(list(used_objs)):  # e.g. used_objs =[1,0,2]
                    obj_name = self.objects[obj]
                    xyz = ycb_poses[index, :]
                    xyz_str = str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2])
                    quat_str = str(xyz[3]) + " " + str(xyz[4]) + " " + str(xyz[5]) + " " + str(xyz[6])
                    if obj == 0:
                        obj_xml = '''
                                         <body name=\"sphere_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"sphere_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.017\" type=\"sphere\"/>
                    </body>          
                    '''
                    elif obj == 1:
                        obj_xml = '''
                                                  <body name=\"cylinder_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"cylinder_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.012 0.045\" type=\"cylinder\"/>
                    </body> 
                                            '''
                    elif obj == 2:
                        obj_xml = '''
                                         <body name=\"box_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"box_1\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
                     </body>
                                         '''
                    elif obj == 3:
                        obj_xml = '''
                                                            <body name=\"box_2\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"box_2\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 1 0.2 1\" size=\"0.015 0.0075 0.03\" type=\"box\"/>
                     </body>
                                                            '''
                    elif obj == 4:
                        obj_xml = '''
                                                                                   <body name=\"box_3\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                                                <geom name=\"box_3\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
                                            </body>
                                                                                   '''
                    else:
                        raise NotImplementedError

                    xml_data += obj_xml

                xml_data += '''
                         </body>
                         </body>
                         '''  # finish hand
                xml_data += '''
                         </worldbody>

                             <actuator>
                 '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>

                         <include file="../descriptions/objs/box.xml"/>

                             </worldbody>

                             <actuator>
                             '''

        #### actuators:
        for i in range(8):
            if self.fingers[i] and i > 1:  # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    xml_data += '''<motor name=\"joint_''' + str(i) + str(
                        j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                 '''
        xml_data += '''</actuator>

         </mujoco>
         '''

        return xml_data


    def return_xml_ycb_real_v3(self, q_grasp=None, q_used=None, ycb_poses=None, color=None, used_objs=None, ):
        # self.palm_height = 0.03
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix_real_v3()
        ii = -1
        ii_max = int(sum(q_used) / 4)
        removed_joints = []
        for i in range(8):  # for each position
            if self.fingers[i]:  # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]  # 4 in default
                x = self.palm_radius * np.cos(theta + np.pi / 2)  # position of the finger start point
                y = self.palm_radius * np.sin(theta + np.pi / 2)
                quat = rot.euler2quat([0, 0, theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])

                # always the first finger used
                if i not in [0]:
                    finger_xml = '''
                                     <body name='finger_''' + str(i) + ''' ' pos=' ''' + str(x) + ' ' + str(
                        y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                 <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                 <body name='MCP_spread_motor_''' + str(i) + '''' pos='0 0 0.0036'>
                                     <inertial pos='0.00780713 0.0116002 -0.00421738' quat='0.556955 0.285031 -0.717518 0.306149' mass='0.218111' diaginertia='7.16804e-05 5.53793e-05 3.8057e-05'/>
                                    <joint name='joint_''' + str(i) + str(0) + '''' type='hinge' pos='0 0 0' axis='0 0 -1' range='-2 2' />
                                     <geom type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                     <geom pos='0 0.009 0' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                     <body name='MCP_motor_''' + str(i) + '''' pos='-0.0135 0.0325 -0.008'>
                                          <inertial pos='0.0133661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                          <joint name='joint_''' + str(i) + str(1) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                         <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                         <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                         <body name='PIP_DIP_motor_''' + str(i) + '''' pos='0 0.049 0'>
                                             <inertial pos='0.0123661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                            <joint name='joint_''' + str(i) + str(2) + '''' pos='0 0 0' axis='-1 0 0' range='-2 2' />
                                             <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                             <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                             <body name='distal_''' + str(i) + ''' ' pos='0 0.049 0'>
                                                 <inertial pos='0.0118444 0.0123626 3.07129e-06' quat='0.503638 0.496337 -0.503565 0.496408' mass='0.163525' diaginertia='3.3711e-05 2.87101e-05 1.70008e-05'/>
                                                 <joint name='joint_''' + str(i) + str(3) + '''' pos='0 0 0' axis='-1 0 0'  range='-2 2'/>
                                                 <geom pos='0.0125 -0.1215 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='DIP_motor_1' name='distal_''' + str(
                        i) + ''''/>
                                                <geom class='sliding_finger' pos='0.0125 -0.1215 0.008' type='mesh' rgba='0 0.7 0 1' mesh='distal_1' name='distal_tip_''' + str(
                        i) + ''''/>
                                                 <site name='finger_''' + str(i) + '''_tip' pos='0.012 0.03 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                             </body>
                                         </body>
                                     </body>
                                 </body>
                          </body>
                                     '''
                else:
                    quat_str_i_j_all = []
                    for j, axis in enumerate(
                            [np.array([0, 0, -1]), np.array([-1, 0, 0]), np.array([-1, 0, 0]), np.array([-1, 0, 0])]):
                        axis_angle = axis * (q_grasp[i * 4 + j])
                        quat_i_j = rot.axisangle2quat(axis_angle)
                        quat_str_i_j = str(quat_i_j[0]) + ' ' + str(quat_i_j[1]) + " " + str(quat_i_j[2]) + " " + str(
                            quat_i_j[3])
                        quat_str_i_j_all.append(quat_str_i_j)
                    finger_xml = '''
                                                         <body name='finger_''' + str(i) + '''' pos=' ''' + str(
                        x) + ' ' + str(y) + '''  0.022' quat =' ''' + quat_str + ''' ' >
                                                     <geom name='base_link_''' + str(i) + '''' type='mesh' rgba='0.7 0.7 0.7 1' mesh='base_link' />
                                                     <body name='MCP_spread_motor_''' + str(
                        i) + ''''  quat=' ''' + quat_str_i_j_all[0] + ''' '   >
                                                          <inertial pos='0.00780713 0.0116002 -0.00421738' quat='0.556955 0.285031 -0.717518 0.306149' mass='0.218111' diaginertia='7.16804e-05 5.53793e-05 3.8057e-05'/>
                                                         <geom type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_spread_motor_1' name='MCP_spread_motor_''' + str(
                        i) + ''''/>
                                                         <geom pos='0 0.009 0' type='mesh' rgba='0.7 0.7 0.7 1' mesh='metacarpal_1' name='metacarpal_''' + str(
                        i) + ''''/>
                                                         <body name='MCP_motor_''' + str(
                        i) + '''' pos='-0.0135 0.0325 -0.008' quat=' ''' + quat_str_i_j_all[1] + ''' '   >
                                                             <inertial pos='0.0133661 0.0158676 3.0672e-06' quat='0.523581 0.475248 -0.523529 0.475309' mass='0.163332' diaginertia='5.41695e-05 4.73107e-05 1.68589e-05'/>
                                                             <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='MCP_motor_1' name='MCP_motor_''' + str(
                        i) + ''''/>
                                                             <geom pos='0.0135 -0.0235 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='proximal_1' name='proximal_''' + str(
                        i) + ''''/>
                                                             <body name='PIP_DIP_motor_''' + str(
                        i) + '''' pos='0.001 0.049 0' quat=' ''' + quat_str_i_j_all[2] + ''' '   >
                                                                 <inertial pos='0.0118444 0.0123626 3.07129e-06' quat='0.503638 0.496337 -0.503565 0.496408' mass='0.163525' diaginertia='3.3711e-05 2.87101e-05 1.70008e-05'/>
                                                                 <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='PIP_DIP_motor_1' name='PIP_DIP_motor_''' + str(
                        i) + '''' />
                                                                 <geom pos='0.0125 -0.0725 0.008' type='mesh' rgba='0.7 0.7 0.7 1' mesh='middle_1' name='middle_''' + str(
                        i) + ''''/>
                                                                 <body name='distal_''' + str(
                        i) + '''' pos='0 0.049 0' quat=' ''' + quat_str_i_j_all[3] + ''' '   >
                                                                     <inertial pos='-0.0135 0.0167496 0' quat='0.5 0.5 -0.5 0.5' mass='0.112487' diaginertia='1.8e-05 1.7e-05 8e-06' />
                                                                     <geom pos='0.0125 -0.1215 0.008' type='mesh' rgba='0.3 0.3 0.3 1' mesh='DIP_motor_1' name='distal_''' + str(
                        i) + ''''/>                                 
                                                                    <geom class='sliding_finger' pos='0.0125 -0.1215 0.008' rgba='0 0.7 0  1' type='mesh' mesh='distal_1' name='distal_tip_''' + str(
                        i) + ''''/>
                                                                     <site name='finger_''' + str(i) + '''_tip' pos='0.012 0.03 0' euler='0 0 0' size='0.005  0.005 0.005'/>
                                                                 </body>
                                                             </body>
                                                         </body>
                                                     </body>
                                              </body>
                                                         '''
                xml_data += finger_xml

        if ycb_poses.shape == (7,):
            ycb_poses = ycb_poses.reshape(1, -1)
        if not self.objects:
            if ycb_poses is not None:
                xml_data += '''</body>
                 <include file="../descriptions/objs/ycb_bodies.xml"/>
                 </worldbody>

                 <actuator>
                 '''
            else:
                xml_data += '''
                 </worldbody>

                 <actuator>
                 '''
        else:  # add objects into xml
            if q_grasp is not None:
                # add spheres which are fixed in the hand base frame
                obj_nums = 3
                rgba_all = []
                for index, obj in enumerate(list(used_objs)):  # e.g. used_objs =[1,0,2]
                    obj_name = self.objects[obj]
                    xyz = ycb_poses[index, :]
                    xyz_str = str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2])
                    quat_str = str(xyz[3]) + " " + str(xyz[4]) + " " + str(xyz[5]) + " " + str(xyz[6])
                    if obj == 0:
                        obj_xml = '''
                                         <body name=\"sphere_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"sphere_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.017\" type=\"sphere\"/>
                    </body>          
                    '''
                    elif obj == 1:
                        obj_xml = '''
                                                  <body name=\"cylinder_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"cylinder_1\" class=\"obj\" mass=\"0.005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.012 0.045\" type=\"cylinder\"/>
                    </body> 
                                            '''
                    elif obj == 2:
                        obj_xml = '''
                                         <body name=\"box_1\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"box_1\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 0 0 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
                     </body>
                                         '''
                    elif obj == 3:
                        obj_xml = '''
                                                            <body name=\"box_2\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                         <geom name=\"box_2\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"1 1 0.2 1\" size=\"0.015 0.0075 0.03\" type=\"box\"/>
                     </body>
                                                            '''
                    elif obj == 4:
                        obj_xml = '''
                                                                                   <body name=\"box_3\" pos=\"''' + xyz_str + ''' \" quat=' ''' + quat_str + ''' '>
                                                <geom name=\"box_3\" class=\"obj\" mass=\"0.00005\" pos=\"0 0 0\" rgba=\"0.92156863 0.81176471 0.69411765 1\" size=\"0.015 0.015 0.015\" type=\"box\"/>
                                            </body>
                                                                                   '''
                    else:
                        raise NotImplementedError

                    xml_data += obj_xml

                xml_data += '''
                         </body>
                         </body>
                         '''  # finish hand
                xml_data += '''
                         </worldbody>

                             <actuator>
                 '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>

                         <include file="../descriptions/objs/box.xml"/>

                             </worldbody>

                             <actuator>
                             '''

        #### actuators:
        for i in range(8):
            if self.fingers[i] and i > 0:  # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    xml_data += '''<motor name=\"joint_''' + str(i) + str(
                        j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                 '''
        xml_data += '''</actuator>

         </mujoco>
         '''

        return xml_data

    def return_xml_ycb(self, q_grasp=None, q_used=None, ycb_poses=None, color=None,used_objs=None):
        # self.palm_height = 0.03
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix(add_ycb=ycb_poses)
        xml_data += '''    <body name=\"hand\" euler=\"0 0 0\" pos=\"0 0 ''' + str(self.palm_height) + '''\">
            <freejoint/>
            '''
        # adding the palm
        xml_data += (
                    "<geom class=\"fingers\" name=\"hand_base\"  pos=\"0 0 0\" quat=\"1 0 0 0\" type=\"cylinder\" size=\"" +
                    str(self.palm_radius) + " " + str(self.palm_height) + "\" rgba=\"0.1 1 .1 1\"/> \n")

        #######   (2) add the fingers
        ii = -1
        ii_max = int(sum(q_used)/4)
        removed_joints = []
        for i in range(8):  # for each position
            if self.fingers[i]:  # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]  # 4 in default
                x = self.palm_radius * np.cos(theta)  # position of the finger start point
                y = self.palm_radius * np.sin(theta)
                quat = rot.euler2quat([0, 0, theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])

                if not self.real:
                    length = self.link_lengths[i]
                    # x1 = length * np.cos(theta)  # position of the first link end
                    # y1 = length * np.sin(theta)
                    x1 = [length] * 4  # position of the first link end
                    y1 = 0
                else:
                    x1 = self.link_lengths[1:]
                    y1 = 0

                axis = [0, 1]  # axis of bending Dof

                if q_used is not None and ii < ii_max and q_used[
                    ii * 4]:  # ii = 0 or 1, and q_used=1, then remove the 1st joint (do not add joints)
                    axis_angle = np.array([0, 0, 1]) * (q_grasp[ii * 4] + theta)
                    quat_i = rot.axisangle2quat(axis_angle)
                    quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(
                        y) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(
                        y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size) + '''\" type=\"capsule\"/>
                        '''
                    removed_joints.append([i, 0])
                    # print([i, 0])

                else:
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(
                        y) + ''' 0\" quat=\"''' + quat_str + '''\">
                            <joint axis=\"0 0 1\" name=\"joint_''' + str(i) + '''0\" pos=\"0 0 0\" range=\"-1 1\" type=\"hinge\"/>
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(
                        y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size) + '''\" type=\"capsule\"/>
                        '''

                rgba_all = [[0.9, 0, 0, 1], [0., 0.6, 0, 1], [0.1, 0.1, 1, 1]]
                for j in range(1, dof_i):  # (1,2,3)
                    rgba = np.array([0.5, 0.5, 0.5, 1])
                    rgba[0] = rgba[0] + 0.1 * j
                    rgba = rgba_all[j - 1]
                    rgba_str = str(rgba[0]) + ' ' + str(rgba[1]) + " " + str(rgba[2]) + " " + str(rgba[3])
                    if q_used is not None and ii < ii_max and q_used[ii * 4 + j]:
                        axis_angle = np.array([0, 1, 0]) * q_grasp[ii * 4 + j]
                        quat_i = rot.axisangle2quat(axis_angle)
                        quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                                        <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(
                            self.capsule_size) + '''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                                    '''

                        removed_joints.append([i, j])
                        # print([i, j])
                    else:
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\">
                              <joint axis=\"''' + str(axis[0]) + " " + str(axis[1]) + ''' 0\" name=\"joint_''' + str(
                            i) + str(j) + '''\" pos=\"0 0 0\" range=\"-2 2\" type=\"hinge\"/>
                              <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(
                            self.capsule_size) + '''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                        '''
                xml_data += '''        <site name="finger_site_''' + str(i) + '''\" pos=\"''' + str(x1[j]) + ''' 0 0\" />
                    '''
                for _ in range(dof_i + 1):
                    xml_data += '''</body>
                    '''

        if not self.objects:
            if ycb_poses is not None:
                xml_data += '''</body>
                <include file="../descriptions/objs/ycb_bodies.xml"/>
                </worldbody>

                <actuator>
                '''
            else:
                xml_data += '''</body>
                </worldbody>

                <actuator>
                '''
        else:  # add objects into xml
            if q_grasp is not None:
                pass
                # add spheres which are fixed in the hand base frame
                obj_nums = 3

                for index, obj in enumerate(list(used_objs)): # e.g. used_objs =[1,0,2]
                    obj_name = self.objects[obj]
                    xyz = ycb_poses[index,:]

                    xml_data += '''<body name=\"''' + obj_name + '''\" pos=\"''' + str(xyz[0]) + " " + str(
                        xyz[1]) + " " + str(xyz[2]) + ''' \">
                                   <geom mesh=\"''' + obj_name + '''\"  material=\"''' +obj_name +'''\" class=\"obj\" type=\"mesh\"/>
                                   </body>                    
                    '''

                xml_data += '''


                            </body>
                        </worldbody>

                            <actuator>
                '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>

                        <include file="../descriptions/objs/box.xml"/>

                            </worldbody>

                            <actuator>
                            '''

        #### actuators:
        for i in range(8):
            if self.fingers[i]:  # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    if [i, j] not in removed_joints:
                        xml_data += '''<motor name=\"joint_''' + str(i) + str(
                            j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                    '''
        xml_data += '''</actuator>

        </mujoco>
        '''

        return xml_data

    def return_xml(self, q_grasp=None, q_used=None, add_ycb=None,color=None,obj_names=None):
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix(add_ycb=add_ycb)
        xml_data += '''    <body name=\"hand\" euler=\"0 0 0\" pos=\"0 0 ''' + str(self.palm_height) + '''\">
            <freejoint/>
            '''
        # adding the palm
        xml_data += ("<geom class=\"fingers\" name=\"hand_base\"  pos=\"0 0 0\" quat=\"1 0 0 0\" type=\"cylinder\" size=\"" +
                     str(self.palm_radius) + " " + str(self.palm_height) + "\" rgba=\"0.1 1 .1 1\"/> \n")

        #######   (2) add the fingers
        ii = -1
        removed_joints = []
        for i in range(8):  # for each possible position to put a finger
            if self.fingers[i]:   # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]   # 4 in default
                x = self.palm_radius * np.cos(theta)  # position of the finger start point
                y = self.palm_radius * np.sin(theta)
                quat = rot.euler2quat([0,0,theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])
                length = self.link_lengths[i]
                # x1 = length * np.cos(theta)  # position of the first link end
                # y1 = length * np.sin(theta)
                if not self.real:
                    length = self.link_lengths[i]
                    # x1 = length * np.cos(theta)  # position of the first link end
                    # y1 = length * np.sin(theta)
                    x1 = [length] * 4  # position of the first link end
                    y1 = 0
                else:
                    x1 = self.link_lengths[1:] # (4,)
                    y1 = 0

                axis = [0, 1]  # axis of bending Dof



                if q_used is not None and ii < 2 and q_used[ii*4]: # ii = 0-2, and q_used=1, then remove the 1st joint (do not add joints)
                    axis_angle = np.array([0,0,1]) * (q_grasp[ii*4] + theta)
                    quat_i = rot.axisangle2quat(axis_angle)
                    quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(y) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size)+'''\" type=\"capsule\"/>
                        '''
                    removed_joints.append([i, 0])
                    # print([i, 0])

                else:
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(y) + ''' 0\" quat=\"''' + quat_str + '''\">
                            <joint axis=\"0 0 1\" name=\"joint_''' + str(i) + '''0\" pos=\"0 0 0\" range=\"-1 1\" type=\"hinge\"/>
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size)+'''\" type=\"capsule\"/>
                        '''

                rgba_all = [[0.9,0,0,1], [0.,0.6,0,1], [0.1,0.1,1,1]]
                for j in range(1, dof_i): #(1,2,3)
                    rgba = np.array([0.5, 0.5, 0.5, 1])
                    rgba[0] = rgba[0] + 0.1 * j
                    rgba = rgba_all[j-1]
                    rgba_str = str(rgba[0]) + ' ' + str(rgba[1]) + " " + str(rgba[2]) + " " + str(rgba[3])
                    if q_used is not None and ii < 2 and q_used[ii * 4 + j]:
                        axis_angle = np.array([0, 1, 0]) * q_grasp[ii * 4 + j]
                        quat_i = rot.axisangle2quat(axis_angle)
                        quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(x1[j-1]) + " " + str(y1) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                                        <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str( y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(self.capsule_size) + '''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                                    '''

                        removed_joints.append([i, j])
                        # print([i, j])
                    else:
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(x1[j-1]) + " " + str(y1) + ''' 0\">
                              <joint axis=\"''' + str(axis[0]) + " " + str(axis[1]) + ''' 0\" name=\"joint_''' + str(i) + str(j) + '''\" pos=\"0 0 0\" range=\"-2 2\" type=\"hinge\"/>
                              <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str(y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(self.capsule_size)+'''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                        '''
                xml_data += '''        <site name="finger_site_''' + str(i) + '''\" pos=\"''' + str(x1[3]) +''' 0 0\" />
                    '''
                for _ in range(dof_i + 1):
                    xml_data += '''</body>
                    '''

        if not self.objects:
            if add_ycb is not None:
                xml_data += '''</body>
                <include file="../descriptions/objs/ycb_bodies.xml"/>
                </worldbody>
                
                <actuator>
                '''
            else:
                xml_data += '''</body>
                </worldbody>
                
                <actuator>
                '''
        else:  # add objects into xml
            if q_grasp is not None:
                pass
                # add spheres which are fixed in the hand base frame
                obj_nums = len(obj_names)
                q_grasp_x = q_grasp[8:]
                sphere_color = [[1, 0, 0, 1], [0, 0, 1, 1], [0.8, 0.8, 0.8,1]]
                sphere_size = [0.02,0.02,0.02]
                for i in range(obj_nums):
                    xyz = q_grasp_x[i*3: i*3+3]
                    sphere_color_i = sphere_color[i]
                    sphere_color_i = str(sphere_color_i[0]) + ' ' + str(sphere_color_i[1]) + " " + str(sphere_color_i[2]) + " " + str(sphere_color_i[3])
                    xml_data += '''<body name=\"sphere_''' + str(i+1) + '''\" pos=\"''' + str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + ''' \">
                                   <geom name=\"sphere_''' + str(i+1) + '''\" class=\"obj\" mass=\"0.005\" rgba=\"'''+ sphere_color_i + '''\" size=\"''' + str(sphere_size[i]) +'''\" type=\"sphere\"/>
                                   </body>                    
                    '''

                xml_data += '''
                            
                            
                            </body>
                        </worldbody>
                        
                            <actuator>
                '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>
                
                        <include file="../descriptions/objs/box.xml"/>
                        
                            </worldbody>
    
                            <actuator>
                            '''

        #### actuators:
        for i in range(8):
            if self.fingers[i]: # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    if [i, j] not in removed_joints:
                        xml_data += '''<motor name=\"joint_''' + str(i) + str(j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                    '''
        xml_data += '''</actuator>
        
        </mujoco>
        '''

        return xml_data

    def return_xml_1_obj(self, q_grasp=None, q_used=None, add_ycb=None, color=None,obj_names=None, fixed_finger=1):
        """

          :return:
        """
        ##  (1) add prefix
        xml_data = self.add_prefix(add_ycb=add_ycb)
        xml_data += '''    <body name=\"hand\" euler=\"0 0 0\" pos=\"0 0 ''' + str(self.palm_height) + '''\">
            <freejoint/>
            '''
        # adding the palm
        xml_data += (
                    "<geom class=\"fingers\" name=\"hand_base\"  pos=\"0 0 0\" quat=\"1 0 0 0\" type=\"cylinder\" size=\"" +
                    str(self.palm_radius) + " " + str(self.palm_height) + "\" rgba=\"0.1 1 .1 1\"/> \n")

        #######   (2) add the fingers
        ii = -1
        removed_joints = []
        for i in range(8):  # for each possible position to put a finger
            if self.fingers[i]:  # if finger exists
                ii += 1
                theta = i * self.d_angle
                dof_i = self.dof[i]  # 4 in default
                x = self.palm_radius * np.cos(theta)  # position of the finger start point
                y = self.palm_radius * np.sin(theta)
                quat = rot.euler2quat([0, 0, theta])
                quat_str = str(quat[0]) + ' ' + str(quat[1]) + " " + str(quat[2]) + " " + str(quat[3])
                length = self.link_lengths[i]
                # x1 = length * np.cos(theta)  # position of the first link end
                # y1 = length * np.sin(theta)
                if not self.real:
                    length = self.link_lengths[i]
                    # x1 = length * np.cos(theta)  # position of the first link end
                    # y1 = length * np.sin(theta)
                    x1 = [length] * 4  # position of the first link end
                    y1 = 0
                else:
                    x1 = self.link_lengths[1:]  # (4,)
                    y1 = 0

                axis = [0, 1]  # axis of bending Dof

                if q_used is not None and ii < fixed_finger and q_used[
                    ii * 4]:  # ii = 0-2, and q_used=1, then remove the 1st joint (do not add joints)
                    axis_angle = np.array([0, 0, 1]) * (q_grasp[ii * 4] + theta)
                    quat_i = rot.axisangle2quat(axis_angle)
                    quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(
                        y) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(
                        y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size) + '''\" type=\"capsule\"/>
                        '''
                    removed_joints.append([i, 0])
                    # print([i, 0])

                else:
                    xml_data += '''
                        <body name=\"finger_''' + str(i) + '''\" pos="0 0 0">
                          <body name=\"aux_''' + str(i) + '''0\" pos=\"''' + str(x) + " " + str(
                        y) + ''' 0\" quat=\"''' + quat_str + '''\">
                            <joint axis=\"0 0 1\" name=\"joint_''' + str(i) + '''0\" pos=\"0 0 0\" range=\"-1 1\" type=\"hinge\"/>
                            <geom class=\"fingers\" fromto=\"0 0 0 ''' + str(x1[0]) + " " + str(
                        y1) + ''' 0\" name="g_''' + str(i) + '''0\" size=\"''' + str(self.capsule_size) + '''\" type=\"capsule\"/>
                        '''

                rgba_all = [[0.9, 0, 0, 1], [0., 0.6, 0, 1], [0.1, 0.1, 1, 1]]
                for j in range(1, dof_i):  # (1,2,3)
                    rgba = np.array([0.5, 0.5, 0.5, 1])
                    rgba[0] = rgba[0] + 0.1 * j
                    rgba = rgba_all[j - 1]
                    rgba_str = str(rgba[0]) + ' ' + str(rgba[1]) + " " + str(rgba[2]) + " " + str(rgba[3])
                    if q_used is not None and ii < fixed_finger and q_used[ii * 4 + j]:
                        axis_angle = np.array([0, 1, 0]) * q_grasp[ii * 4 + j]
                        quat_i = rot.axisangle2quat(axis_angle)
                        quat_str_i = str(quat_i[0]) + ' ' + str(quat_i[1]) + " " + str(quat_i[2]) + " " + str(quat_i[3])
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(
                            x1[j - 1]) + " " + str(y1) + ''' 0\" quat=\"''' + quat_str_i + '''\">
                                        <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(
                            self.capsule_size) + '''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                                    '''

                        removed_joints.append([i, j])
                        # print([i, j])
                    else:
                        xml_data += '''<body name=\"aux_''' + str(i) + str(j) + '''\" pos=\"''' + str(
                            x1[j - 1]) + " " + str(y1) + ''' 0\">
                              <joint axis=\"''' + str(axis[0]) + " " + str(axis[1]) + ''' 0\" name=\"joint_''' + str(
                            i) + str(j) + '''\" pos=\"0 0 0\" range=\"-2 2\" type=\"hinge\"/>
                              <geom class=\"fingers\"  fromto=\"0 0 0 ''' + str(x1[j]) + " " + str(
                            y1) + ''' 0\" name="g_''' + str(i) + str(j) + '''\" size=\"''' + str(
                            self.capsule_size) + '''\" type=\"capsule\" rgba=\"''' + rgba_str + '''\"/>
                        '''
                xml_data += '''        <site name="finger_site_''' + str(i) + '''\" pos=\"''' + str(x1[3]) + ''' 0 0\" />
                    '''
                for _ in range(dof_i + 1):
                    xml_data += '''</body>
                    '''

        if not self.objects:
            if add_ycb is not None:
                xml_data += '''</body>
                <include file="../descriptions/objs/ycb_bodies.xml"/>
                </worldbody>

                <actuator>
                '''
            else:
                xml_data += '''</body>
                </worldbody>

                <actuator>
                '''
        else:  # add objects into xml
            if q_grasp is not None:
                pass
                # add spheres which are fixed in the hand base frame
                obj_nums = len(obj_names)
                q_grasp_x = q_grasp[fixed_finger*4:]
                sphere_color = [[1, 0, 0, 1], [0, 0, 1, 1], [0.8, 0.8, 0.8, 1]]
                sphere_size = [0.02, 0.02, 0.02]
                for i in range(obj_nums):
                    xyz = q_grasp_x[i * 3: i * 3 + 3]
                    sphere_color_i = sphere_color[i]
                    sphere_color_i = str(sphere_color_i[0]) + ' ' + str(sphere_color_i[1]) + " " + str(
                        sphere_color_i[2]) + " " + str(sphere_color_i[3])
                    xml_data += '''<body name=\"sphere_''' + str(i + 1) + '''\" pos=\"''' + str(xyz[0]) + " " + str(
                        xyz[1]) + " " + str(xyz[2]) + ''' \">
                                   <geom name=\"sphere_''' + str(
                        i + 1) + '''\" class=\"obj\" mass=\"0.005\" rgba=\"''' + sphere_color_i + '''\" size=\"''' + str(
                        sphere_size[i]) + '''\" type=\"sphere\"/>
                                   </body>                    
                    '''

                xml_data += '''


                            </body>
                        </worldbody>

                            <actuator>
                '''

            else:
                # add spheres with freejoint
                xml_data += '''</body>

                        <include file="../descriptions/objs/box.xml"/>

                            </worldbody>

                            <actuator>
                            '''

        #### actuators:
        for i in range(8):
            if self.fingers[i]:  # for each finger
                dof_i = self.dof[i]
                for j in range(dof_i):
                    if [i, j] not in removed_joints:
                        xml_data += '''<motor name=\"joint_''' + str(i) + str(
                            j) + '''\" ctrllimited=\"true\" ctrlrange=\"-2 2\" joint=\"joint_''' + str(i) + str(j) + '''\"/>
                    '''
        xml_data += '''</actuator>

        </mujoco>
        '''

        return xml_data

    def add_prefix_real_v3(self):
        a1 = """
        <mujoco model='iiwa7'>
    <compiler angle='radian' inertiafromgeom='true' meshdir='../descriptions/v2/meshes/'/>
    <size njmax='500' nconmax='100' />

    <option>
        <flag gravity='enable' />
    </option>

    <default>
        <joint limited='true' damping='1' armature='0'/>
        <geom contype='1' conaffinity='1' condim='3' rgba='0.8 0.6 .4 1'
        	margin='0.001' solref='.02 1' solimp='.8 .8 .01' material='geom'/>
    </default>
<visual>
       <scale forcewidth = "0.1"  contactwidth = "0.1" contactheight = "0.1" />
 </visual>
 
 <default>
    <joint damping='0.01' frictionloss='0'/>
    <default class='visual'>
      <geom contype='0' conaffinity='0'/>
    </default>
    <default class='collision'>
      <geom contype='1' conaffinity='1' friction='1 0 0.1' solref='-100000 -200'  />
    </default>
      <default class='obj'>
      <geom condim='6' friction='1 0.5 0.1'  />
    </default>

      <default class='fingers'>
      <geom friction='1 0.5 0.2'  />
    </default>

      <default class='sliding_finger'>
      <geom friction='0.001 0. 0'  />
    </default>
      <default class='visual1'>
      <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual2'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual3'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual4'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual5'>
      <geom contype='1' conaffinity='1'/>
    </default>

    <site rgba='1 0 0 0'/>
  </default>

        <asset>
            <mesh name='base_link' file='base_link.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_spread_motor_1' file='MCP_spread_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_motor_1' file='MCP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='proximal_1' file='proximal_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='PIP_DIP_motor_1' file='PIP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='middle_1' file='middle_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='distal_1' file='distal_1.stl' scale='0.001 0.001 0.001' />
             <mesh name="DIP_motor_1" file="DIP_motor_1.stl" scale="0.001 0.001 0.001"/>
            <mesh name='metacarpal_1' file='metacarpal_1.stl' scale='0.001 0.001 0.001' />
<!--		<texture type='skybox' builtin='gradient' width='128' height='128' rgb1='.4 .6 .8'-->
<!--            rgb2='0 0 0'/>-->
            <texture name='texgeom' type='cube' builtin='flat' mark='cross' width='127' height='1278'
                rgb1='0.8 0.6 0.4' rgb2='0.8 0.6 0.4' markrgb='1 1 1' random='0.01'/>
            <texture name='texplane' type='2d' builtin='checker' rgb1='.2 .3 .4' rgb2='.1 0.15 0.2'
                width='512' height='512'/>

            <material name='MatPlane' reflectance='0.5' texture='texplane' texrepeat='1 1' texuniform='true'/>
            <material name='geom' texture='texgeom' texuniform='true'/>
             <texture builtin='flat' name='allegro_mount_tex' height='32' width='32' rgb1='0.2 0.2 0.2' type='cube'/>
             <texture builtin='flat' name='hand_tex' height='32' width='32' rgb1='0.2 0.2 0.2 ' type='cube'/>
             <texture builtin='flat' name='tip_tex' height='32' width='32' rgb1='0.9 0.9 0.9' type='cube'/>

            <material name='allegro_mount_mat' shininess='0.03' specular='0.25'/>
            <material name='hand_mat' shininess='0.03' specular='0.25' />
            <material name='tip_mat' shininess='0.03' specular='0.25' />
        </asset>

    <worldbody>
        <light pos='0 0 10' castshadow='false'/>
         <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
        <body name='world_base' euler='0 0 0' pos='0 0 -0.05'>
            <geom name='floor' class='collision' type='plane' size='10 10 1' rgba='.8 .8 .8 1' material='MatPlane'/>
        </body>

        <body name='hand_base' euler='0 0 0' pos='0 0 0.0'>
            <freejoint/>
             <geom class='fingers' name='hand_base_1' mass='0.01'  pos='0 0 0.03' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='.95 0.95 0.95 1'/>
             <geom class='fingers' name='hand_base_2' mass='0.01' pos='0 0 0' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='.95 0.95 0.95 1'/>
            <body name='hand'>
        """

        return a1

    def add_prefix_real(self):
        a1 = """
        <mujoco model='iiwa7'>
    <compiler angle='radian' inertiafromgeom='true' meshdir='../descriptions/single_finger/shortest/meshes/'/>
    <size njmax='500' nconmax='100' />

    <option>
        <flag gravity='enable' />
    </option>

    <default>
        <joint limited='true' damping='1' armature='0'/>
        <geom contype='1' conaffinity='1' condim='3' rgba='0.8 0.6 .4 1'
            margin='0.001' solref='.02 1' solimp='.8 .8 .01' material='geom'/>
    </default>
<visual>
       <scale forcewidth = "0.1"  contactwidth = "0.1" contactheight = "0.1" />
 </visual>

 <default>
    <joint damping='0.01' frictionloss='0'/>
    <default class='visual'>
      <geom contype='0' conaffinity='0'/>
    </default>
    <default class='collision'>
      <geom contype='1' conaffinity='1' friction='1 0 0.1' solref='-100000 -200'  />
    </default>
      <default class='obj'>
      <geom condim='6' friction='1 0.5 0.1'  />
    </default>

      <default class='fingers'>
      <geom friction='1 0.5 0.2'  />
    </default>

      <default class='sliding_finger'>
      <geom friction='0.001 0. 0'  />
    </default>
      <default class='visual1'>
      <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual2'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual3'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual4'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual5'>
      <geom contype='1' conaffinity='1'/>
    </default>

    <site rgba='1 0 0 0'/>
  </default>

        <asset>
            <mesh name='base_link' file='base_link.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_spread_motor_1' file='MCP_spread_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_motor_1' file='MCP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='proximal_1' file='proximal_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='PIP_DIP_motor_1' file='PIP_DIP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='middle_1' file='middle_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='distal_1' file='distal_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='metacarpal_1' file='metacarpal_1.stl' scale='0.001 0.001 0.001' />
<!--		<texture type='skybox' builtin='gradient' width='128' height='128' rgb1='.4 .6 .8'-->
<!--            rgb2='0 0 0'/>-->
            <texture name='texgeom' type='cube' builtin='flat' mark='cross' width='127' height='1278'
                rgb1='0.8 0.6 0.4' rgb2='0.8 0.6 0.4' markrgb='1 1 1' random='0.01'/>
            <texture name='texplane' type='2d' builtin='checker' rgb1='.2 .3 .4' rgb2='.1 0.15 0.2'
                width='512' height='512'/>

            <material name='MatPlane' reflectance='0.5' texture='texplane' texrepeat='1 1' texuniform='true'/>
            <material name='geom' texture='texgeom' texuniform='true'/>
             <texture builtin='flat' name='allegro_mount_tex' height='32' width='32' rgb1='0.2 0.2 0.2' type='cube'/>
             <texture builtin='flat' name='hand_tex' height='32' width='32' rgb1='0.2 0.2 0.2 ' type='cube'/>
             <texture builtin='flat' name='tip_tex' height='32' width='32' rgb1='0.9 0.9 0.9' type='cube'/>

            <material name='allegro_mount_mat' shininess='0.03' specular='0.25'/>
            <material name='hand_mat' shininess='0.03' specular='0.25' />
            <material name='tip_mat' shininess='0.03' specular='0.25' />
        </asset>

    <worldbody>
        <light pos='0 0 10' castshadow='false'/>
         <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
        <body name='world_base' euler='0 0 0' pos='0 0 -0.05'>
            <geom name='floor' class='collision' type='plane' size='10 10 1' rgba='.8 .8 .8 1' material='MatPlane'/>
        </body>

        <body name='hand_base' euler='0 0 0' pos='0 0 0.0'>
            <freejoint/>
             <geom class='fingers' name='hand_base_1' mass='0.01'  pos='0 0 0.03' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.1 1 .1 1'/>
             <geom class='fingers' name='hand_base_2' mass='0.01' pos='0 0 0' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.1 1 .1 1'/>
            <body name='hand'>
        """

        return a1

    def add_prefix_real_v2(self):
        a1 = """
        <mujoco model='iiwa7'>
    <compiler angle='radian' inertiafromgeom='true' meshdir='../descriptions/v2/meshes/'/>
    <size njmax='500' nconmax='100' />

    <option>
        <flag gravity='enable' />
    </option>

    <default>
        <joint limited='true' damping='1' armature='0'/>
        <geom contype='1' conaffinity='1' condim='3' rgba='0.8 0.6 .4 1'
            margin='0.001' solref='.02 1' solimp='.8 .8 .01' material='geom'/>
    </default>
<visual>
       <scale forcewidth = "0.1"  contactwidth = "0.1" contactheight = "0.1" />
 </visual>

 <default>
    <joint damping='0.01' frictionloss='0'/>
    <default class='visual'>
      <geom contype='0' conaffinity='0'/>
    </default>
    <default class='collision'>
      <geom contype='1' conaffinity='1' friction='1 0 0.1' solref='-100000 -200'  />
    </default>
      <default class='obj'>
      <geom condim='6' friction='1 0.5 0.1'  />
    </default>

      <default class='fingers'>
      <geom friction='1 0.5 0.2'  />
    </default>

      <default class='sliding_finger'>
      <geom friction='0.001 0. 0'  />
    </default>
      <default class='visual1'>
      <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual2'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual3'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual4'>
     <geom contype='1' conaffinity='1'/>
    </default>
        <default class='visual5'>
      <geom contype='1' conaffinity='1'/>
    </default>

    <site rgba='1 0 0 0'/>
  </default>

        <asset>
            <mesh name='base_link' file='base_link.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_spread_motor_1' file='MCP_spread_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='MCP_motor_1' file='MCP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='proximal_1' file='proximal_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='PIP_DIP_motor_1' file='PIP_motor_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='middle_1' file='middle_1.stl' scale='0.001 0.001 0.001' />
            <mesh name='distal_1' file='distal_1.stl' scale='0.001 0.001 0.001' />
             <mesh name="DIP_motor_1" file="DIP_motor_1.stl" scale="0.001 0.001 0.001"/>
            <mesh name='metacarpal_1' file='metacarpal_1.stl' scale='0.001 0.001 0.001' />
<!--		<texture type='skybox' builtin='gradient' width='128' height='128' rgb1='.4 .6 .8'-->
<!--            rgb2='0 0 0'/>-->
            <texture name='texgeom' type='cube' builtin='flat' mark='cross' width='127' height='1278'
                rgb1='0.8 0.6 0.4' rgb2='0.8 0.6 0.4' markrgb='1 1 1' random='0.01'/>
            <texture name='texplane' type='2d' builtin='checker' rgb1='.2 .3 .4' rgb2='.1 0.15 0.2'
                width='512' height='512'/>

            <material name='MatPlane' reflectance='0.5' texture='texplane' texrepeat='1 1' texuniform='true'/>
            <material name='geom' texture='texgeom' texuniform='true'/>
             <texture builtin='flat' name='allegro_mount_tex' height='32' width='32' rgb1='0.2 0.2 0.2' type='cube'/>
             <texture builtin='flat' name='hand_tex' height='32' width='32' rgb1='0.2 0.2 0.2 ' type='cube'/>
             <texture builtin='flat' name='tip_tex' height='32' width='32' rgb1='0.9 0.9 0.9' type='cube'/>

            <material name='allegro_mount_mat' shininess='0.03' specular='0.25'/>
            <material name='hand_mat' shininess='0.03' specular='0.25' />
            <material name='tip_mat' shininess='0.03' specular='0.25' />
        </asset>

    <worldbody>
        <light pos='0 0 10' castshadow='false'/>
         <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
        <body name='world_base' euler='0 0 0' pos='0 0 -0.05'>
            <geom name='floor' class='collision' type='plane' size='10 10 1' rgba='.8 .8 .8 1' material='MatPlane'/>
        </body>

        <body name='hand_base' euler='0 0 0' pos='0 0 0.0'>
            <freejoint/>
             <geom class='fingers' name='hand_base_1' mass='0.01'  pos='0 0 0.03' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.95 0.95 0.95 1'/>
             <geom class='fingers' name='hand_base_2' mass='0.01' pos='0 0 0' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.95 0.95 0.95 1'/>
            <body name='hand'>
        """

        return a1

        def add_prefix_real(self):
            a1 = """
            <mujoco model='iiwa7'>
        <compiler angle='radian' inertiafromgeom='true' meshdir='../descriptions/single_finger/shortest/meshes/'/>
        <size njmax='500' nconmax='100' />

        <option>
            <flag gravity='enable' />
        </option>

        <default>
            <joint limited='true' damping='1' armature='0'/>
            <geom contype='1' conaffinity='1' condim='3' rgba='0.8 0.6 .4 1'
                margin='0.001' solref='.02 1' solimp='.8 .8 .01' material='geom'/>
        </default>
    <visual>
           <scale forcewidth = "0.1"  contactwidth = "0.1" contactheight = "0.1" />
     </visual>

     <default>
        <joint damping='0.01' frictionloss='0'/>
        <default class='visual'>
          <geom contype='0' conaffinity='0'/>
        </default>
        <default class='collision'>
          <geom contype='1' conaffinity='1' friction='1 0 0.1' solref='-100000 -200'  />
        </default>
          <default class='obj'>
          <geom condim='6' friction='1 0.5 0.1'  />
        </default>

          <default class='fingers'>
          <geom friction='1 0.5 0.2'  />
        </default>

          <default class='sliding_finger'>
          <geom friction='0.001 0. 0'  />
        </default>
          <default class='visual1'>
          <geom contype='1' conaffinity='1'/>
        </default>
            <default class='visual2'>
         <geom contype='1' conaffinity='1'/>
        </default>
            <default class='visual3'>
         <geom contype='1' conaffinity='1'/>
        </default>
            <default class='visual4'>
         <geom contype='1' conaffinity='1'/>
        </default>
            <default class='visual5'>
          <geom contype='1' conaffinity='1'/>
        </default>

        <site rgba='1 0 0 0'/>
      </default>

            <asset>
                <mesh name='base_link' file='base_link.stl' scale='0.001 0.001 0.001' />
                <mesh name='MCP_spread_motor_1' file='MCP_spread_motor_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='MCP_motor_1' file='MCP_motor_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='proximal_1' file='proximal_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='PIP_DIP_motor_1' file='PIP_DIP_motor_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='middle_1' file='middle_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='distal_1' file='distal_1.stl' scale='0.001 0.001 0.001' />
                <mesh name='metacarpal_1' file='metacarpal_1.stl' scale='0.001 0.001 0.001' />
    <!--		<texture type='skybox' builtin='gradient' width='128' height='128' rgb1='.4 .6 .8'-->
    <!--            rgb2='0 0 0'/>-->
                <texture name='texgeom' type='cube' builtin='flat' mark='cross' width='127' height='1278'
                    rgb1='0.8 0.6 0.4' rgb2='0.8 0.6 0.4' markrgb='1 1 1' random='0.01'/>
                <texture name='texplane' type='2d' builtin='checker' rgb1='.2 .3 .4' rgb2='.1 0.15 0.2'
                    width='512' height='512'/>

                <material name='MatPlane' reflectance='0.5' texture='texplane' texrepeat='1 1' texuniform='true'/>
                <material name='geom' texture='texgeom' texuniform='true'/>
                 <texture builtin='flat' name='allegro_mount_tex' height='32' width='32' rgb1='0.2 0.2 0.2' type='cube'/>
                 <texture builtin='flat' name='hand_tex' height='32' width='32' rgb1='0.2 0.2 0.2 ' type='cube'/>
                 <texture builtin='flat' name='tip_tex' height='32' width='32' rgb1='0.9 0.9 0.9' type='cube'/>

                <material name='allegro_mount_mat' shininess='0.03' specular='0.25'/>
                <material name='hand_mat' shininess='0.03' specular='0.25' />
                <material name='tip_mat' shininess='0.03' specular='0.25' />
            </asset>

        <worldbody>
            <light pos='0 0 10' castshadow='false'/>
             <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
            <body name='world_base' euler='0 0 0' pos='0 0 -0.05'>
                <geom name='floor' class='collision' type='plane' size='10 10 1' rgba='.8 .8 .8 1' material='MatPlane'/>
            </body>

            <body name='hand_base' euler='0 0 0' pos='0 0 0.0'>
                <freejoint/>
                 <geom class='fingers' name='hand_base_1' mass='0.01'  pos='0 0 0.03' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.1 1 .1 1'/>
                 <geom class='fingers' name='hand_base_2' mass='0.01' pos='0 0 0' quat='1 0 0 0' type='cylinder' size='0.08 0.001' rgba='0.1 1 .1 1'/>
                <body name='hand'>
            """

            return a1

    def add_prefix(self, add_ycb=None):
        """
                    Basic xml string for MuJoCo simulation
                    :return:
                    """
        if add_ycb is not None:
            a_ycb = """
            <include file="../descriptions/objs/ycb_preloads.xml"/>
            """
        else:
            a_ycb = ''

        a1 = """<mujoco model="hand">
            <compiler angle="radian" assetdir="../descriptions/" inertiafromgeom="true" />
            <size njmax="500" nconmax="100" />
            
            <visual>
              <global offwidth="1920" offheight="1080"/>
            </visual>
            
            <option>
                <flag gravity="enable" />
            </option>

            <default>
                <joint limited='true' damping='1' armature='0'/>
                <geom contype='1' conaffinity='1' condim='3' rgba='0.8 0.6 .4 1'
                    margin="0.001" solref=".02 1" solimp=".8 .8 .01" material="geom"/>
            </default>

         <default>
            <joint damping="0.01" frictionloss="0"/>
            <default class="visual">
              <geom contype="0" conaffinity="0"/>
            </default>
            <default class="collision">
              <geom contype="1" conaffinity="1" friction="1 0.5 0.1" />
            </default>
              <default class="obj">
              <geom condim="3" priority="1" friction="1 1 1" solref=".02 0.8" solimp=".8 .8 .01"/>
            </default>

              <default class="fingers">
              <geom friction="1 0.5 0.2"  />
            </default>
            <site rgba="1 0 0 0" size="0.005 0.005 0.005"/>
          </default>

                <asset>
                    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278"
                        rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
                    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                        width="512" height="512"/>

                    <material name='MatPlane' reflectance='0.5' texture="texplane" texrepeat="1 1" texuniform="true"/>
                    <material name='geom' texture="texgeom" texuniform="true"/>
                     <texture builtin="flat" name="allegro_mount_tex" height="32" width="32" rgb1="0.2 0.2 0.2" type="cube"/>
                     <texture builtin="flat" name="hand_tex" height="32" width="32" rgb1="0.2 0.2 0.2 " type="cube"/>
                     <texture builtin="flat" name="tip_tex" height="32" width="32" rgb1="0.9 0.9 0.9" type="cube"/>

                    <material name="allegro_mount_mat" shininess="0.03" specular="0.25"/>
                    <material name="hand_mat" shininess="0.03" specular="0.25" />
                    <material name="tip_mat" shininess="0.03" specular="0.25" />
                </asset>
    """
        a2 = """
            <worldbody>
                <light pos="0 0 1000" castshadow="false"/>
                 <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 -.15 -1'/>
                 <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
                <geom name="floor" class="collision" type="plane" size="10 10 1" rgba=".8 .8 .8 1" pos='0 0 -0.25' material="MatPlane"/>
                """
        return a1 + a_ycb + a2

    def add_objects(self):
        """
        adding objects into the xml file, for example, spheres and cubes
        :return:  xml string
        """
        pass
