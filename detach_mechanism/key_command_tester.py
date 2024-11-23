from detach_mechanism.comms_wrapper import Arduino, Key
import time


class detach_control():
    def __init__(self):
        self.arduino = Arduino("motor controller", "/dev/ttyACM0", 115200)
        self.arduino.connect_and_handshake()

    def run(self, i=1, t_total=5):
        """
        detach from the iiwa
        :param i:  i =1, detach, i=0, attach
        :return:
        """
        t0 = time.time()
        while 1:
            cmd = [0, 0]
            if time.time() - t0 < t_total:
                if i == 1:
                    cmd[1] = 1
                elif i == 0:
                    cmd[1] = -1
                self.arduino.send_message(cmd)
            else:
                self.arduino.send_message(cmd)
                break


            time.sleep(0.01)


if __name__ == "__main__":
    key = Key()

    arduino = Arduino("motor controller", "/dev/ttyACM0", 115200)
    arduino.connect_and_handshake()

    while 1:
        cmd = [0, 0]
        if key.keyPress == "w":
            cmd[0] = 1
        elif key.keyPress == "s":
            cmd[0] = -1
        elif key.keyPress == "d":
            cmd[1] = 1
        elif key.keyPress == "a":
            cmd[1] = -1

        arduino.send_message(cmd)

        # arduino.receive_message()
        # print(arduino.receivedMessages)

        time.sleep(0.01)