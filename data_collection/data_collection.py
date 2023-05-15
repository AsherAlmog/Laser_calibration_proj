import serial
import threading
import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import picamera
import os

def capture_pic(x,z,theta,preview_flag, root_dir):
    with picamera.PiCamera() as camera:
        # camera.resolution = resolution
        if(preview_flag):
            camera.start_preview()
        time.sleep(1)  # wait for the camera to warm up
        img_name = os.path.join(root_dir, f"{x},{z},{theta}.jpg")
        camera.capture(img_name)
        if (preview_flag):
            camera.stop_preview()
        print(f"Captured image {x},{z},{theta}: {img_name}")


def flip_dir(dir):
    if (dir==1):
        return 0
    if(dir == 0):
        return 1

def decide_dir(num):
    if num < 0:
        return 1
    else:
        return 0

baud_rate = 9600
port = 'COM5'
ser = serial.Serial(port, baud_rate)  # , timeout=10)
# Clear the input and output buffers of the serial port
ser.flushInput()
ser.flushOutput()
print("starting the comm")
coordinates_dict = {'x': 0, 'z': 0, 'theta': 0, 'dirz': 0, 'dirx':1, 'dirtz':0}
pic_dict = {'pic': 'taken'}
ack_msg = ""
ack_msg = ser.readline().decode().strip()
while ack_msg != "ready":
    ack_msg = ser.readline().decode().strip()
print(f"ack_msg is {ack_msg}")
ack_msg = ""
# the vectors are organized in this order: [x,z,theta_z]
bottom_bound = [8,6,1]
upper_bound = [9,7,2]
range_arr = [1, 1, 1]

root_dir = "C:/Users/asher/PycharmProjects/Laser_calibration_proj/speckles_pic"

for x in range(bottom_bound[0], upper_bound[0]+1):
    for z in range(bottom_bound[1], upper_bound[1] + 1):
        for theta in range(bottom_bound[2], upper_bound[2] + 1):
            coordinates_dict['z'] = abs(z)
            coordinates_dict['x'] = abs(x)
            coordinates_dict['theta'] = abs(theta)
            coordinates_dict['dirz'] = flip_dir(curr_dir_z)
            coordinates_dict['dirx'] = flip_dir(curr_dir_x)
            coordinates_dict['dirtz'] = flip_dir(curr_dir_tz)

            json_str = json.dumps(coordinates_dict)+'\n'
            json_str = json_str.encode('utf-8')
            ser.write(json_str)
            print(f"Sent json {coordinates_dict} to arduino")

            # wait for a message from arduino that he moved to position
            ack_msg = ser.readline().decode().strip()
            while ack_msg != "moved":
                ack_msg = ser.readline().decode().strip()
            print(f"ack_msg is {ack_msg}")
            ack_msg = ""
            # now we should take a pic
            capture_pic(x, z, theta, 0, root_dir)

            # now tell the arduino to turn the other way by flipping dir
            curr_dir_z = coordinates_dict['dirz']
            curr_dir_x = coordinates_dict['dirx']
            curr_dir_tz = coordinates_dict['dirtz']

            coordinates_dict['dirz'] = flip_dir(curr_dir_z)
            coordinates_dict['dirx'] = flip_dir(curr_dir_x)
            coordinates_dict['dirtz'] = flip_dir(curr_dir_tz)

            json_str = json.dumps(coordinates_dict) + '\n'
            json_str = json_str.encode('utf-8')
            ser.write(json_str)
            print(f"Sent json {coordinates_dict} flipped to arduino")

            ack_msg = ser.readline().decode().strip()
            while ack_msg != "moved":
                ack_msg = ser.readline().decode().strip()
            print(f"ack_msg after flipping is {ack_msg}")
            ack_msg = ""

            # now flip back the direction



