# MIT License

# Copyright (c) 2022 Haoran Sun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np

from dynamixel_sdk import *

class DynamixelController:
    def __init__(self, port_name, motor_list, 
        protocol=2.0, baudrate=3000000, latency_time=1, reverse_direction=False) -> None:
        '''
            Args:
                port_name: name of the usb port (/dev/ttyUSB0 for example)
                motor_list: a list of motors objects
                protocol: protocol version
                baudrate: baudrate of the communication bus
                latency_time: latency time in milliseconds to be set for 
                    the USB device, if set to 0, latency time will not be set.
        '''
        
        self.port_name = port_name
        self.__device_name = self.port_name.split('/')[2]
        self.motor_list = motor_list
        self.protocol = protocol
        self.baudrate = baudrate
        self.latency_time = latency_time
        self.reverse_direction = reverse_direction

        self._port_handler = PortHandler(self.port_name)
        self._packet_handler = PacketHandler(self.protocol)

        self.__motor_model = self.motor_list[0]
        self.__motor_ids = []

        self.__sync_writer_registers = {
            "drive_mode": self.__motor_model.drive_mode,
            "operating_mode": self.__motor_model.operating_mode,            
            
            "torque_enable": self.__motor_model.torque_enable,
            "led": self.__motor_model.led,
            
            "velocity_i_gain": self.__motor_model.velocity_i_gain,
            "velocity_p_gain": self.__motor_model.velocity_p_gain,
            "return_delay_time": self.__motor_model.return_delay_time,

            "position_d_gain": self.__motor_model.position_d_gain,
            "position_i_gain": self.__motor_model.position_i_gain,
            "position_p_gain": self.__motor_model.position_p_gain,
            "feedforward_2nd_gain": self.__motor_model.feedforward_2nd_gain,
            "feedforward_1st_gain": self.__motor_model.feedforward_1st_gain,

            "goal_pwm": self.__motor_model.goal_pwm,
            "goal_current": self.__motor_model.goal_current,
            "goal_velocity": self.__motor_model.goal_velocity,
            "profile_acceleration": self.__motor_model.profile_acceleration,
            "profile_velocity": self.__motor_model.profile_velocity,
            "goal_position": self.__motor_model.goal_position,
        }

        self.__sync_writers = dict()

        self.__bulk_info_reader = None

        # Conversion scales
        self.__position_deg_scale = 0.087891 # per encoder value
        self.__position_rad_scale = (self.__position_deg_scale/180.0)*np.pi
        self.__velocity_deg_scale = (0.229*360.0)/60.0 # degree/s
        self.__velocity_rad_scale = (self.__velocity_deg_scale/180.0)*np.pi
        self.__current_mA_scale = 2.69
        self.__pwm_percent_scale = 0.11299
        self.__max_voltage = 12.0

    def offset_origin(self, encoder_list):
        if not self.reverse_direction:
            return (encoder_list - 2048)
        else:
            # Reserse mode
            return (2048 - encoder_list)

    def offset_degree_encoder(self, degree_encoder):
        if not self.reverse_direction:
            return degree_encoder + 2048
        else:
            return 2048 - degree_encoder

    def activate_controller(self):
        # Config motor list
        for motor in self.motor_list:
            self.__motor_ids.append(motor.motor_id)

        if not self._port_handler.openPort():
            raise PortCommError("Failed to open port {}. ".format(self.port_name))
        
        if not self._port_handler.setBaudRate(self.baudrate):
            raise PortCommError("Failed to set baud rate to {}. ".format(self.baudrate))

        # set usb latency time
        # if self.latency_time != 0:
        #     print("Setting USB latency time to {} for device {}. ".format(self.latency_time, self.__device_name))
        #     cmd = "echo {} | sudo tee \
        #         /sys/bus/usb-serial/devices/{}/latency_timer".format(self.latency_time, self.__device_name)
        #     os.system(cmd)

        for motor_id in self.__motor_ids:
            dxl_model_number, dxl_comm_result, dxl_error = self._packet_handler.ping(self._port_handler, motor_id)
            if dxl_comm_result != COMM_SUCCESS:
                raise PortCommError("Failed to ping motor of id {}. ".format(motor_id))
            elif dxl_error != 0:
                raise PortCommError("Failed to ping motor of id {}. dxl_error: {}. ".format(motor_id, 
                    self._packet_handler.getRxPacketError(dxl_error)))
            else:
                pass
                # print("[DynamixelController] ID:%03d ping succeeded. Dynamixel model number : %d" % (motor_id, dxl_model_number))

        # init sync writers
        self.__init_sync_writers()

        # init info reader
        start_addr_info_reader = self.__motor_model.present_pwm.address
        len_info_reader = self.__motor_model.present_pwm.size + \
                          self.__motor_model.present_current.size + \
                          self.__motor_model.present_velocity.size + \
                          self.__motor_model.present_position.size
        self.__bulk_info_reader = GroupSyncRead(self._port_handler,
                                         self._packet_handler,
                                         start_addr_info_reader,
                                         len_info_reader)
        for motor_id in self.__motor_ids:
            self.__bulk_info_reader.addParam(motor_id)

    def __init_sync_writers(self):
        '''
            initialize sync writers.

            self.__sync_writers = {name: GroupSyncWrite, ...}
        '''
        for sync_writer_name in self.__sync_writer_registers.keys():
            register = self.__sync_writer_registers[sync_writer_name]
            register_addr = register.address
            register_size = register.size
            self.__sync_writers[sync_writer_name] = GroupSyncWrite(self._port_handler,
                                                                   self._packet_handler,
                                                                   register_addr,
                                                                   register_size)

    def __sync_write(self, sync_writer_name, data_list):
        '''
            Args:
                sync_writer_name: the name string of the writer
                data_list: list of data of form [data1, data2, ...] to be written;
                    must be in the same shape of motor_list
        '''
        # if type(sync_writer_name) == list:
        #     for j in range(len(sync_writer_name)):
        self.__sync_writers[sync_writer_name].clearParam()
        sync_write_size = self.__sync_writer_registers[sync_writer_name].size

        assert len(self.__motor_ids) == len(data_list), \
            "Length of the data {} is not consistent with length of motor list {}. ".format(len(self.motor_list), len(data_list))

        for i, motor_id in enumerate(self.__motor_ids):
            if sync_write_size == 1:
                written_value = [data_list[i]]
            elif sync_write_size == 2:
                written_value = [DXL_LOBYTE(DXL_LOWORD(data_list[i])),
                                 DXL_HIBYTE(DXL_LOWORD(data_list[i]))]
            elif sync_write_size == 4:
                written_value = [DXL_LOBYTE(DXL_LOWORD(data_list[i])),
                                 DXL_HIBYTE(DXL_LOWORD(data_list[i])),
                                 DXL_LOBYTE(DXL_HIWORD(data_list[i])),
                                 DXL_HIBYTE(DXL_HIWORD(data_list[i]))]

            self.__sync_writers[sync_writer_name].addParam(motor_id, written_value)

        return self.__sync_writers[sync_writer_name].txPacket()

    def torque_enable(self, torque_enable_list):
        '''
            Args:
                torque_enable_list: a list of torque_enable value, 0 for disable, 1 for enable
        '''
        return self.__sync_write("torque_enable", torque_enable_list)
    
    def torque_on(self):
        data_list = [0x1] * len(self.__motor_ids)
        return self.torque_enable(data_list)
    
    def torque_off(self):
        data_list = [0x0] * len(self.__motor_ids)
        return self.torque_enable(data_list)

    def led_enable(self, led_enable_list):
        '''
            Args;
                led_enable_list: a list of led value, 0 for disable, 1 for enable
        '''
        return self.__sync_write("led", led_enable_list)

    def led_on(self):
        '''
            Enable all leds
        '''
        led_enable_list = [0x1] * len(self.__motor_ids)
        return self.led_enable(led_enable_list)

    def led_off(self):
        '''
            Disable all leds
        '''
        led_enable_list = [0x0] * len(self.__motor_ids)
        return self.led_enable(led_enable_list)

    def set_goal_current(self, current_list):
        return self.__sync_write("goal_current", current_list)

    def set_return_delay_time(self, delay_time):
        t = np.round(delay_time).astype(int)
        return self.__sync_write("return_delay_time", t)

    def set_goal_current_mA(self, current_list):
        np_current_list = np.array(current_list)
        encoder_current_list = np.round(np_current_list/self.__current_mA_scale).astype(int)
        return self.set_goal_current(encoder_current_list)

    def set_goal_position(self, position_list):
        return self.__sync_write("goal_position", position_list)

    def set_goal_position_deg(self, position_list):
        np_position_list = np.array(position_list)
        encoder_position_list = np.round(self.offset_degree_encoder(np_position_list/self.__position_deg_scale)).astype(int)
        return self.set_goal_position(encoder_position_list)

    def set_goal_position_rad(self, position_list):
        np_position_list = np.array(position_list)
        encoder_position_list = np.round(self.offset_degree_encoder(np_position_list/self.__position_rad_scale)).astype(int)
        return self.set_goal_position(encoder_position_list)

    def set_drive_mode(self, mode_list):
        return self.__sync_write('drive_mode', mode_list)

    def set_operating_mode(self, op_list):
        return self.__sync_write("operating_mode", op_list)

    def set_operating_mode_all(self, operating_mode:str):
        '''
            Add the motors' operating mode will be set to the one specified by
            arg operating_mode.

            operating_mode can be one of the following (in string):
                (0) current_control
                (1) velocity_control
                (3) position_control
                (4) extended_position_control
                (5) current_based_position_control
                (16) pwm_control
        '''
        if operating_mode == "current_control":
            op_list = [0x0] * len(self.__motor_ids)
        elif operating_mode == "velocity_control":
            op_list = [0x1] * len(self.__motor_ids)
        elif operating_mode == "position_control":
            op_list = [0x3] * len(self.__motor_ids)
        elif operating_mode == "extended_position_control":
            op_list = [0x4] * len(self.__motor_ids)
        elif operating_mode == "current_based_position_control":
            op_list = [0x5] * len(self.__motor_ids)
        elif operating_mode == "pwm_control":
            op_list = [0x10] * len(self.__motor_ids)
        else:
            raise ValueError(operating_mode)

        return self.set_operating_mode(op_list)

    def set_velocity_i_gain(self, vi_gain_list):
        return self.__sync_write("velocity_i_gain", vi_gain_list)

    def set_velocity_p_gain(self, vp_gain_list):
        return self.__sync_write("velocity_p_gain", vp_gain_list)

    def set_position_d_gain(self, pos_d_gain_list):
        return self.__sync_write("position_d_gain", pos_d_gain_list)

    def set_position_i_gain(self, pos_i_gain_list):
        return self.__sync_write("position_i_gain", pos_i_gain_list)

    def set_position_p_gain(self, pos_p_gain_list):
        return self.__sync_write("position_p_gain", np.int16(pos_p_gain_list))

    def set_feedforward_2nd_gain(self, ff_2nd_gain_list):
        return self.__sync_write("feedforward_2nd_gain", ff_2nd_gain_list)

    def set_feedforward_1st_gain(self, ff_1st_gain_list):
        return self.__sync_write("feedforward_1st_gain", ff_1st_gain_list)
    
    def set_goal_pwm(self, pwm_list):
        return self.__sync_write("goal_pwm", pwm_list)

    def set_goal_velocity(self, velocity_list):
        if self.reverse_direction:
            velocity_list = -velocity_list
        return self.__sync_write("goal_velocity", velocity_list)

    def set_goal_velocity_deg(self, velocity_list):
        np_vel_list = np.array(velocity_list)
        encoder_vel_list = np.round(np_vel_list/self.__velocity_deg_scale).astype(int)
        return self.set_goal_velocity(encoder_vel_list)

    def set_goal_velocity_rad(self, velocity_list):
        np_vel_list = np.array(velocity_list)
        encoder_vel_list = np.round(np_vel_list/self.__velocity_rad_scale).astype(int)
        return self.set_goal_velocity(encoder_vel_list)

    def set_profile_acceleration(self, acc_list):
        encoder_acc_list = np.round(acc_list / self.__current_mA_scale).astype(int)
        return self.__sync_write("profile_acceleration", encoder_acc_list)

    def set_profile_velocity(self, p_vel_list):
        return self.__sync_write("profile_velocity", p_vel_list)

    def set_profile_velocity_deg(self, p_vel_list):
        np_vel_list = np.array(p_vel_list)
        encoder_vel_list = np.round(np_vel_list/self.__velocity_deg_scale).astype(int)
        return self.set_profile_velocity(encoder_vel_list)

    def set_profile_velocity_rad(self, p_vel_list):
        np_vel_list = np.array(p_vel_list)
        encoder_vel_list = np.round(np_vel_list/self.__velocity_rad_scale).astype(int)
        return self.set_profile_velocity(encoder_vel_list)

    def read_info(self, retry=True, max_retry_time=3, fast_read=True):
        '''
            Args:
                retry: whether or not retry reading from the bus in case of faults like packet loss
                max_retry_time: maximum retry time;

            Read the present position, velocity and current for 
            all the motors;

            We did not use the default api to parse the read data, because the 
            original Dynamixel SDK does not support parsing data composed of multiple
            memory blocks;
            Instead we read data_dict, which will be in the form of {motor_id: [byte1, byte2, ...], ...}

            return type:
            (position_list, velocity_list, current_list)
        '''
        if fast_read:
            dxl_comm_result = self.__bulk_info_reader.fastSyncRead()
        else:
            dxl_comm_result = self.__bulk_info_reader.txRxPacket()
            data_arrays = list(self.__bulk_info_reader.data_dict.values())

            # In case of communication error, read again
            if dxl_comm_result != COMM_SUCCESS and retry:
                retry_time = 0
                while dxl_comm_result != COMM_SUCCESS and retry_time < max_retry_time:
                    #print(dxl_comm_result)
                    dxl_comm_result = self.__bulk_info_reader.txRxPacket()
                    data_arrays = list(self.__bulk_info_reader.data_dict.values())
                    retry_time += 1

        # If still cannot read normally
        if dxl_comm_result != COMM_SUCCESS:
            raise PortCommError("Packet reading error while trying to read information from the bus. ")

        if fast_read:
            pwm_list = []
            current_list = []
            velocity_list = []
            position_list = []
            for motor_id in self.__motor_ids:
                dxl_getdata_result = self.__bulk_info_reader.isAvailable(
                    motor_id,
                    self.__motor_model.present_pwm.address,
                    self.__motor_model.present_pwm.size + \
                    self.__motor_model.present_current.size + \
                    self.__motor_model.present_velocity.size + \
                    self.__motor_model.present_position.size
                )
                if not dxl_getdata_result:
                    raise PortCommError("Failed to fast sync read data from motor of id {}. ".format(motor_id))
                pwm_list.append(self.__bulk_info_reader.getData(motor_id, self.__motor_model.present_pwm.address, self.__motor_model.present_pwm.size))
                current_list.append(self.__bulk_info_reader.getData(motor_id, self.__motor_model.present_current.address, self.__motor_model.present_current.size))
                velocity_list.append(self.__bulk_info_reader.getData(motor_id, self.__motor_model.present_velocity.address, self.__motor_model.present_velocity.size))
                position_list.append(self.__bulk_info_reader.getData(motor_id, self.__motor_model.present_position.address, self.__motor_model.present_position.size))

            # Convert to numpy
            pwm_list = np.array(pwm_list)
            current_list = np.array(current_list)
            velocity_list = np.array(velocity_list)
            position_list = np.array(position_list)
        else:
            data_stack = np.stack(data_arrays) # Size (num_motors, 10)

            pwm_list = DXL_MAKEWORD(data_stack[:, 0], data_stack[:, 1])
            current_list = DXL_MAKEWORD(data_stack[:, 2], data_stack[:, 3])
            velocity_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 4], data_stack[:, 5]),
                                        DXL_MAKEWORD(data_stack[:, 6], data_stack[:, 7]))
            position_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 8], data_stack[:, 9]),
                                        DXL_MAKEWORD(data_stack[:, 10], data_stack[:, 11]))

        # handle negative values
        offset_vel_list = (velocity_list > 0x7fffffff).astype(int) * 4294967296
        velocity_list -= offset_vel_list
        offset_cur_list = (current_list > 0x7fff).astype(int) * 65536
        current_list -= offset_cur_list
        pwm_list -= offset_cur_list

        if self.reverse_direction:
            velocity_list = -velocity_list

        return (position_list, velocity_list, current_list, pwm_list)

    def read_info_with_unit(self, pwm_unit="percent", angle_unit="rad", current_unit="mA", retry=True, max_retry_time=3, fast_read=True):
        '''
            Args:
                pwm_unit: the following units are accepted:
                    (1) "percent": percentage of the PWM
                    (2) "vol": effective voltage
                    (3) "raw": raw pwm register value
                angle_unit: the following units are accepted:
                    (1) "rad": rad for angle, rad/s for angular velocity
                    (2) "deg": degree for angle, degree/s for angular velocity
                current_unit: the following units are accepted:
                    (1) "mA": mA for current value
                    (2) "raw": raw current register value
                retry: whether or not retry reading from the bus in case of faults like packet loss
                max_retry_time: maximum retry time;
        '''
        position_list, velocity_list, current_list, pwm_list = self.read_info(retry, max_retry_time, fast_read=fast_read)
        position_list = self.offset_origin(position_list) # transform the origin
        if pwm_unit == "percent":
            pwm_list = pwm_list * self.__pwm_percent_scale
        elif pwm_unit == "vol":
            pwm_list = pwm_list * self.__pwm_percent_scale * self.__max_voltage / 100.0
        if angle_unit == "rad":
            position_list = position_list * self.__position_rad_scale
            velocity_list = velocity_list * self.__velocity_rad_scale
        elif angle_unit == "deg":
            position_list = position_list * self.__position_deg_scale
            velocity_list = velocity_list * self.__velocity_deg_scale
        
        if current_unit == "mA":
            current_list = current_list * self.__current_mA_scale

        return position_list, velocity_list, current_list, pwm_list

class PortCommError(Exception):
    def __init__(self, msg) -> None:
        self.msg = msg
    
    def __str__(self) -> str:
        return self.msg