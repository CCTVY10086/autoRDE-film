import jkrc
import math
import time
import ADP
from datetime import datetime
import json
def process_experiment(data):
    print("parameter_1:", data["parameter_1"])
    print("parameter_2:", data["parameter_2"])
    print("parameter_3:", data["parameter_3"])
    print("parameter_4:", data["parameter_4"])
    print("parameter_5:", data["parameter_5"])
with open('experiment_params/data.json', 'r', encoding='utf-8') as file:
    experiment_data = json.load(file)
solution_ratio = experiment_data['parameter_4']
additive_content = experiment_data['parameter_3']
def d2r(a):
    return a / 180.0 * math.pi
INCR = 1
robot = jkrc.RC("192.168.193.5")
start = [7, 375, 450.0, d2r(180), d2r(29), d2r(-90)]
stop_0 = [-163.5, 418, 108.0, d2r(180), d2r(0), d2r(-90)]
stop_1 = [-172.5, 418, 108.0, d2r(180), d2r(0), d2r(-90)]
point_0 = [84, 423.5, 280.0, d2r(180), d2r(0), d2r(-90)]
point_1 = [-50, 450, 270.0, d2r(180), d2r(0), d2r(-90)]
point_2 = [-100, 450, 243.0, d2r(180), d2r(0), d2r(-90)]
throw = [-111.5, 550.0, 400, d2r(180), d2r(0), d2r(-90)]
right_30 = [-30, 0, 0, 0, 0, 0]
down_15 = [0, 0, -15, 0, 0, 0]
down_143 = [0, 0, -143, 0, 0, 0]
down_135 = [0, 0, -135, 0, 0, 0]
up_150 = [0, 0, 150, 0, 0, 0]
stop_y = -9
sip_y = -22
def ink(ratio, add, total_volume=1000):
    water_v = math.ceil(total_volume * ratio)
    ybc_v = math.ceil((1 - ratio) * total_volume)
    add_v = math.ceil(add * total_volume * 0.01)
    return water_v, ybc_v, add_v
water, ybc, additive = ink(solution_ratio, additive_content, 1000)
pipette = ADP.ElectronicPipette(port='COM6')
pipette.initialize()
robot.login()
robot.enable_robot()
robot.linear_move(start, 0, True, 70)
robot.linear_move(right_30, INCR, True, 20)
def aspirate_multi(volume, max_volume=1000):
    num_full = volume // max_volume
    remainder = volume % max_volume

    for i in range(int(num_full)):
        print(f" {i + 1} time：{max_volume} μL")
        pipette.aspirate(max_volume)

    if remainder > 0:
        print(f"{num_full + 1} time：{remainder} μL")
        pipette.aspirate(remainder)
def water_volume(volume, max_volume=1000):
    num_full = volume // max_volume
    remainder = volume % max_volume
    robot.linear_move(stop_0, 0, True, 40)
    time.sleep(1)
    robot.linear_move(down_15, INCR, True, 50)
    robot.linear_move(up_150, INCR, True, 70)
    for i in range(int(num_full)):
        robot.linear_move(point_2, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f"{i + 1} time：{max_volume} μL")
        pipette.aspirate(max_volume)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(max_volume)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
    if remainder > 0:
        robot.linear_move(point_2, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f"{num_full + 1} time：{remainder} μL")
        pipette.aspirate(remainder)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(remainder)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
        robot.linear_move(throw, 0, True, 70)
        pipette.eject_tip()
def ybc_volume(volume, max_volume=1000):
    num_full = volume // max_volume
    remainder = volume % max_volume
    robot.linear_move(stop_1, 0, True, 40)
    time.sleep(1)
    robot.linear_move(down_15, INCR, True, 50)
    robot.linear_move(up_150, INCR, True, 70)
    for i in range(int(num_full)):
        robot.linear_move(point_1, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f"{i + 1} time：{max_volume} μL")
        pipette.aspirate(max_volume)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(max_volume)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
    if remainder > 0:
        robot.linear_move(point_1, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f"{num_full + 1} time：{remainder} μL")
        pipette.aspirate(remainder)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(remainder)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
        robot.linear_move(throw, 0, True, 70)
        pipette.eject_tip()
def add_volume(volume, max_volume=1000):
    num_full = volume // max_volume
    remainder = volume % max_volume
    robot.linear_move(stop_1, 0, True, 40)
    time.sleep(1)
    robot.linear_move(down_15, INCR, True, 50)
    robot.linear_move(up_150, INCR, True, 70)
    for i in range(int(num_full)):
        robot.linear_move(point_1, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f" {i + 1} time：{max_volume} μL")
        pipette.aspirate(max_volume)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(max_volume)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
    if remainder > 0:
        robot.linear_move(point_1, 0, True, 40)
        robot.linear_move(down_143, INCR, True, 50)
        print(f"{num_full + 1} time：{remainder} μL")
        pipette.aspirate(remainder)
        time.sleep(20)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(point_0, 0, True, 40)
        robot.linear_move(down_135, INCR, True, 50)
        pipette.dispense(remainder)
        time.sleep(15)
        robot.linear_move(up_150, INCR, True, 70)
        robot.linear_move(start, 0, True, 70)
        robot.linear_move(throw, 0, True, 70)
        pipette.eject_tip()
for x_count in range(1):
    pipette.eject_tip()
    water_volume(water, 1000)
    ybc_volume(ybc, 1000)
    add_volume(additive, 50)
    stop_0[1] += stop_y
    stop_1[1] += stop_y
    point_0[1] += sip_y
