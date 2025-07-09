import jkrc
import math
import motor
import time
import ADP
import json
import serial
from datetime import datetime
from typing import Optional


def process_experiment(data):
    print("parameter_1:", data["parameter_1"])
    print("parameter_2:", data["parameter_2"])
    print("parameter_3:", data["parameter_3"])
    print("parameter_4:", data["parameter_4"])
    print("parameter_5:", data["parameter_5"])


with open('experiment_params/data.json', 'r', encoding='utf-8') as file:
    experiment_data = json.load(file)

drying_speed = experiment_data['parameter_3']
ser = serial.Serial('COM7', 115200, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS,
                    stopbits=serial.STOPBITS_ONE)
# 电子移液枪
ser_1 = serial.Serial('COM6', 9600, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE)
# 电机
# 夹爪
init = [0x01, 0x06, 0x01, 0x00, 0x00, 0xA5, 0x48, 0x4D]
force = [0x01, 0x06, 0x01, 0x01, 0x00, 0x48, 0x58, 0x04]
k = [0x01, 0x06, 0x01, 0x03, 0x03, 0xE8, 0x78, 0x88]
b = [0x01, 0x06, 0x01, 0x03, 0x00, 0x00, 0x78, 0x36]
run = [0x02, 0x31, 0x31, 0x57, 0x52, 0x03, 0x04]
v_max = [0x02, 0x31, 0x31, 0x56, 0x31, 0x30, 0x2C, 0x31, 0x52, 0x03, 0x19]
esc = [0x02, 0x31, 0x31, 0x45, 0x31, 0x52, 0x03, 0x27]
P = [0x02, 0x31, 0x31, 0x50, 0x32, 0x30, 0x2C, 0x31, 0x52, 0x03, 0x1C]
D = [0x02, 0x31, 0x31, 0x44, 0x31, 0x32, 0x2C, 0x31, 0x52, 0x03, 0x09]
v = [0x02, 0x31, 0x31, 0x76, 0x32, 0x2C, 0x31, 0x52, 0x03, 0x0A]
ser_1.write(bytes(run))
time.sleep(1)
ser.write(bytes(v_max))
time.sleep(1)
ser.write(bytes(v))
INCR = 1
robot = jkrc.RC("192.168.193.5")
log_file = open('log.txt', 'w')


def d2r(a):
    return a / 180.0 * math.pi


electrode_z_values = {
    1: 257.6,
    2: 256.3,
    3: 256.3,
    4: 256.3,
    5: 257.5,
    6: 256.4,
    7: 257.2,
    8: 256.8,
    9: 257.3,
}


def get_z_value(electrode_id):
    return electrode_z_values.get(electrode_id, None)


try:
    user_input = input("number_int: ")
    electrode_id = int(user_input)
    z_value = get_z_value(electrode_id)

except ValueError:
    print("error")

start = [7, 375, 450.0, d2r(180), d2r(29), d2r(-90)]
stop = [-110, 418, 108.0, d2r(180), d2r(0), d2r(-90)]
sip_point = [84, 423.5, 232.0, d2r(180), d2r(0), d2r(-90)]
'''drip_point = [-32.5, 369.5, 256.75, d2r(180), d2r(0), d2r(-90)]'''
drip_point = [-32.9, 369.2, z_value, d2r(180), d2r(0), d2r(-90)]
end = [-33.7, 369.5, 241.3, d2r(180), d2r(0), d2r(-90)]
mid = [-32.7, 368.5, 241.3, d2r(180), d2r(0), d2r(-90)]
throw = [-111.5, 550.0, 244, d2r(180), d2r(0), d2r(-90)]
grip_point = [38, 385.5, 85, d2r(180), d2r(29), d2r(-90)]
put_point = [-34.6, 436.15, 178.0, d2r(180), d2r(29), d2r(-90)]
down_50 = [0, 0, -50, 0, 0, 0]
down_17 = [0, 0, -17, 0, 0, 0]
up_100 = [0, 0, 100, 0, 0, 0]
down_15 = [0, 0, -15, 0, 0, 0]
up_2 = [0, 0, 2, 0, 0, 0]
down_2 = [0, 0, -2, 0, 0, 0]
up_150 = [0, 0, 150, 0, 0, 0]
down_150 = [0, 0, -150, 0, 0, 0]
right_10 = [-30, 0, 0, 0, 0, 0]
right_1 = [-1, 0, 0, 0, 0, 0]
stop_y = -9
stop_x = -9
sip_y = -22
sip_z = -0.1
grip_y = 20


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


try:
    with open('y_count.txt', 'r') as file:
        y_count = int(file.read().strip())
except FileNotFoundError:
    y_count = 0

stop[1] += y_count * stop_y
sip_point[2] += y_count * sip_z


def get_electrode():
    ser.write(bytes(k))
    robot.linear_move(start, 0, True, 70)
    robot.linear_move(grip_point, 0, True, 70)
    robot.linear_move(down_50, INCR, True, 20)
    ser.write(bytes(b))
    print(f"[{get_current_time()}] get")
    time.sleep(1)


def put_electrode(speed=100):
    controller.soft_stop(decel_time=1000)
    robot.linear_move(up_100, INCR, True, 70)
    robot.linear_move(up_150, INCR, True, 70)
    robot.linear_move(put_point, 0, True, 70)
    controller.set_rotation(speed, motor.Direction.REVERSE)
    time.sleep(0.1)
    robot.linear_move(down_17, INCR, True, 20)
    time.sleep(5)
    ser.write(bytes(k))
    controller.soft_stop(decel_time=1000)


def get_ink():
    robot.linear_move(up_150, INCR, True, 70)
    robot.linear_move(stop, 0, True, 40)
    time.sleep(1)
    robot.linear_move(down_15, INCR, True, 50)
    robot.linear_move(up_150, INCR, True, 70)
    robot.linear_move(sip_point, 0, True, 40)
    robot.linear_move(down_150, INCR, True, 50)
    time.sleep(0.5)
    ser_1.write(bytes(P))
    time.sleep(5)


def drip_ink(speed=100, drying_speed=400):
    controller.emergency_stop()
    robot.linear_move(up_150, INCR, True, 70)
    robot.linear_move(drip_point, 0, True, 40)
    robot.linear_move(down_15, INCR, True, 30)
    time.sleep(0.5)
    controller.set_rotation(speed, motor.Direction.REVERSE)
    ser_1.write(bytes(D))
    time.sleep(3)
    robot.linear_move(up_2, INCR, True, 1)
    robot.linear_move(right_10, INCR, True, 10)
    controller.set_rotation(drying_speed, motor.Direction.REVERSE)
    robot.linear_move(throw, 0, True, 40)
    ser_1.write(bytes(esc))


def detach_electrode(speed=100):
    ser.write(bytes(k))
    if controller.is_device_ready():
        controller.soft_stop()
    else:
        print("Unresponded")
    controller.soft_stop()
    robot.linear_move(put_point, 0, True, 30)
    controller.set_rotation(speed, motor.Direction.FORWARD)  # 电机正转
    robot.linear_move(down_17, INCR, True, 20)
    ser.write(bytes(b))
    print(f"[{get_current_time()}] put")
    time.sleep(3)
    robot.linear_move(up_150, INCR, True, 70)
    robot.linear_move(grip_point, 0, True, 50)
    robot.linear_move(down_50, INCR, True, 20)
    controller.soft_stop()
    ser.write(bytes(k))


controller = motor.RotateController(port='COM8')
robot.login()
robot.enable_robot()
for x_count in range(1):
    get_electrode()
    put_electrode(150)
    get_ink()
    drip_ink(100, drying_speed)
    time.sleep(10)
    detach_electrode(200)
    stop[0] += stop_x
    sip_point[1] += sip_y
    grip_point[1] += grip_y
    robot.linear_move(start, 0, True, 70)
if y_count < 11:
    y_count += 1
else:
    y_count = 0
with open('y_count.txt', 'w') as file:
    file.write(str(y_count))
