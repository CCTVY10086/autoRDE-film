import jkrc
import time
import math
import serial

ser_1 = serial.Serial('COM4', 115200, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS,
                      stopbits=serial.STOPBITS_ONE)

ser_2 = serial.Serial('COM7', 115200, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS,
                      stopbits=serial.STOPBITS_ONE)

ser_3 = serial.Serial('COM3', 9600, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS,
                      stopbits=serial.STOPBITS_ONE)

ser_4 = serial.Serial('COM5', 115200, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS,
                      stopbits=serial.STOPBITS_ONE)

init = [0x01, 0x06, 0x01, 0x00, 0x00, 0xA5, 0x48, 0x4D]
force = [0x01, 0x06, 0x01, 0x01, 0x00, 0x48, 0x58, 0x04]
k = [0x01, 0x06, 0x01, 0x03, 0x03, 0xE8, 0x78, 0x88]
b = [0x01, 0x06, 0x01, 0x03, 0x00, 0x00, 0x78, 0x36]

start = [0xFF, 0x06, 0x00, 0x04, 0x00, 0x01, 0x1C, 0x15]
stop_1 = [0xFF, 0x06, 0x00, 0x04, 0x00, 0x00, 0xDD, 0xD5]

v = [0x01, 0x06, 0x00, 0x9A, 0x00, 0x0A, 0x29, 0xE2]
change = [0x01, 0x10, 0x00, 0xDE, 0x00, 0x02, 0x04, 0x07, 0xD0, 0x00, 0x00, 0x7F, 0xF2]

power = [0x01, 0xF3, 0xAB, 0x01, 0x00, 0x6B]
work = [0x01, 0xF6, 0x01, 0x01, 0x2C, 0x0A, 0x00, 0x6B]
stop_2 = [0x01, 0xFE, 0x98, 0x00, 0x6B]
ser_2.write(bytes(init))
time.sleep(1)
ser_2.write(bytes(force))
ser_1.write(bytes(v))
time.sleep(2)


def d2r(a):
    return a / 180.0 * math.pi


ser_2.write(bytes(k))
rc = jkrc.RC("192.168.193.5")
start_1 = [290, 375, 500, d2r(-179.5), d2r(29), d2r(-90)]
grip_pose = [524.17, 367.54, 17, d2r(-179.5), d2r(29), d2r(-90)]
clear_pos = [400.23, 285, -15, d2r(-179.5), d2r(29), d2r(-90)]
clear_pos_1 = [400.23, 245, -5, d2r(-179.5), d2r(60), d2r(-90)]
change_pose = [316.5, 301.0, 118, d2r(-179.5), d2r(29), d2r(-180)]
mid_pose = [319, 264, 510, d2r(-179.5), d2r(29), d2r(-90)]
grip_point = [38, 385.5, 90, d2r(180), d2r(29), d2r(-90)]
stop_pose = [180.00, 425.00, 350.00, d2r(-179.5), d2r(29), d2r(-90)]
start_pose = [180.00, 425.00, 293.00, d2r(-179.5), d2r(29), d2r(-90)]
end_pose_1 = [176.00, 429.00, 293.00, d2r(-179.5), d2r(29), d2r(-90)]
mid_pose_1 = [184.00, 429.00, 293.00, d2r(-179.5), d2r(29), d2r(-90)]
mid_pose_2 = [184.00, 421.00, 293.00, d2r(-179.5), d2r(29), d2r(-90)]
end_pose_2 = [176.00, 421.00, 293.00, d2r(-179.5), d2r(29), d2r(-90)]
down_100 = [0, 0, -100, 0, 0, 0]
up_400 = [0, 0, 400, 0, 0, 0]
down_50 = [0, 0, -50, 0, 0, 0]
up_100_50 = [0, -50, 100, 0, 0, 0]
up_10 = [0, 0, 10, 0, 0, 0]
up_100 = [0, 0, 100, 0, 0, 0]
right_200 = [-200, 0, 0, 0, 0, 0]
left_100 = [100, 0, 0, 0, 0, 0]
left_8 = [10, 0, 0, 0, 0, 0]
right_8 = [-10, 0, 0, 0, 0, 0]
up_5 = [0, 0, 10, 0, 0, 0]
down_5 = [0, 0, -10, 0, 0, 0]
grip_y = 20
y = -23.5
'''rc.login()
rc.linear_move(start_1, 0, True, 90)
rc.linear_move(grip_pose, 0, True, 90)
rc.linear_move(down_100, 1, True, 40)
ser_2.write(bytes(b))
time.sleep(1)
rc.linear_move(up_400, 1, True, 90)
rc.linear_move(stop_pose, 0, True, 90)
rc.linear_move(start_pose, 0, True, 10)
time.sleep(1)
for w in range(50):
    rc.circular_move_extend(end_pose_1, mid_pose_1, 0, True, 10, 20, 0, None, 1)
    rc.circular_move_extend(end_pose_2, mid_pose_2, 0, False, 10, 20, 0, None, 1)
rc.linear_move(stop_pose, 0, True, 90)
rc.linear_move(up_100_50, 1, True, 90)'''
for i in range(6):
    rc.login()
    rc.linear_move(start_1, 0, True, 90)
    rc.linear_move(grip_pose, 0, True, 90)
    rc.linear_move(down_100, 1, True, 40)
    ser_2.write(bytes(b))
    time.sleep(1)
    rc.linear_move(up_400, 1, True, 90)
    rc.linear_move(stop_pose, 0, True, 90)
    rc.linear_move(start_pose, 0, True, 10)
    time.sleep(1)
    for w in range(50):
        rc.circular_move_extend(end_pose_1, mid_pose_1, 0, True, 30, 20, 0, None, 1)
        rc.circular_move_extend(end_pose_2, mid_pose_2, 0, False, 30, 20, 0, None, 1)
    rc.linear_move(stop_pose, 0, True, 90)
    rc.linear_move(up_100_50, 1, True, 90)
    rc.linear_move(left_100, 1, True, 90)
    rc.linear_move(clear_pos, 0, True, 90)
    rc.linear_move(down_100, 1, True, 40)
    ser_3.write(bytes(start))
    time.sleep(30)
    ser_3.write(bytes(stop_1))
    '''rc.linear_move(clear_pos_1, 0, True, 90)
    rc.linear_move(down_100, 1, True, 40)'''
    '''ser_4.write(bytes(work))'''
    '''for q in range(8):
        rc.linear_move(left_8, 1, True, 30)
        rc.linear_move(up_5, 1, True, 30)
        rc.linear_move(right_8, 1, True, 30)
        rc.linear_move(down_5, 1, True, 30)
    ser_4.write(bytes(stop_2))'''
    time.sleep(0)
    rc.linear_move(up_400, 1, True, 90)
    rc.linear_move(change_pose, 0, True, 90)
    rc.linear_move(down_100, 1, True, 25)
    time.sleep(0.5)
    ser_2.write(bytes(k))
    rc.linear_move(up_100, 1, True, 70)
    ser_1.write(bytes(change))
    time.sleep(2)
    rc.linear_move(change_pose, 0, True, 90)
    rc.linear_move(down_100, 1, True, 25)
    ser_2.write(bytes(b))
    time.sleep(1)
    rc.linear_move(up_400, 1, True, 90)
    rc.linear_move(mid_pose, 0, True, 90)
    rc.linear_move(right_200, 1, True, 90)
    rc.linear_move(grip_point, 0, True, 90)
    rc.linear_move(down_50, 1, True, 40)
    time.sleep(0.5)
    ser_2.write(bytes(k))
    rc.linear_move(grip_point, 0, True, 40)
    rc.linear_move(up_400, 1, True, 90)
    rc.linear_move(mid_pose, 0, True, 90)
    rc.logout()
    grip_point[1] += grip_y
    grip_pose[1] += y
