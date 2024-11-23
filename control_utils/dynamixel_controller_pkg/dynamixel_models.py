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

from collections import namedtuple

class Register:
    def __init__(self, address:int, size:int,
        EEPROM:bool=False, read_only:bool=False) -> None:
        self.address = address
        self.size = size
        self.EEPROM = EEPROM
        self.read_only = read_only

class BaseModel:
    # EEPROM area
    model_number = Register(0, 2, True, True)
    model_information = Register(2, 4, True)
    firmware_version = Register(6, 1, True)
    id = Register(7, 1, True)
    baud_rate = Register(8, 1, True)
    return_delay_time = Register(9, 1, True)
    drive_mode = Register(10, 1, True)
    operating_mode = Register(11, 1, True)
    secondary_id = Register(12, 1, True)
    protocol_type = Register(13, 1, True)
    homing_offset = Register(20, 4, True)
    moving_threshold = Register(24, 4, True)
    temperature_limit = Register(31, 1, True)
    max_voltage_limit = Register(32, 2, True)
    min_voltage_limit = Register(34, 2, True)
    pwm_limit = Register(36, 2, True)
    current_limit = Register(38, 2, True)
    velocity_limit = Register(44, 4, True)
    max_position_limit = Register(48, 4, True)
    min_position_limit = Register(52, 4, True)
    startup_configuration = Register(60, 1, True)
    shutdown = Register(63, 1, True)

    # RAM area
    torque_enable = Register(64, 1)
    led = Register(65, 1)

    status_return_level = Register(68, 1)
    registered_instruction = Register(69, 1, False, True)
    hardware_error_status = Register(70, 1, False, True)

    velocity_i_gain = Register(76, 2)
    velocity_p_gain = Register(78, 2)

    position_d_gain = Register(80, 2)
    position_i_gain = Register(82, 2)
    position_p_gain = Register(84, 2)

    feedforward_2nd_gain = Register(88, 2)
    feedforward_1st_gain = Register(90, 2)
    
    bus_watchdog = Register(98, 1)

    goal_pwm = Register(100, 2)
    goal_current = Register(102, 2)
    goal_velocity = Register(104, 4)
    profile_acceleration = Register(108, 4)
    profile_velocity = Register(112, 4)
    goal_position = Register(116, 4)

    realtime_tick = Register(120, 2, False, True)
    moving = Register(122, 1, False, True)
    moving_status = Register(123, 1, False, True)
    present_pwm = Register(124, 2, False, True)
    present_current = Register(126, 2, False, True)
    present_velocity = Register(128, 4, False, True)
    present_position = Register(132, 4, False, True)
    velocity_trajectory = Register(136, 4, False, True)
    position_trajectory = Register(140, 4, False, True)
    present_input_voltage = Register(144, 2, False, True)
    present_temperature = Register(146, 1, False, True)
    backup_ready = Register(147, 1, False, True)
    
    def __init__(self, motor_id=1) -> None:
        self.set_motor_id(motor_id)

    def set_motor_id(self, motor_id):
        assert isinstance(motor_id, int) and \
            motor_id >= 0 and motor_id <= 252, \
            "Motor ID {} not accepted".format(motor_id)
        if not hasattr(self, "motor_id"):
            # Motor id not set yet
            setattr(self, 'motor_id', motor_id)
        else:
            self.motor_id = motor_id

    def __str__(self) -> str:
        str_info = "Model name: {}; Motor ID: {}. ".format(self.model_name, self.motor_id)
        return str_info

class XM430(BaseModel):
    def __init__(self, motor_id=1) -> None:
        super().__init__(motor_id)
        self.model_name = "XM430"

class XM430W210(XM430):
    def __init__(self, motor_id=1) -> None:
        super().__init__(motor_id)
        self.model_name = "XM430-W210"

class XM430W350(XM430):
    def __init__(self, motor_id=1) -> None:
        super().__init__(motor_id)
        self.model_name = "XM430-W350"




    