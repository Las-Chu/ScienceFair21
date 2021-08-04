
########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
import numpy as np
 
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

from enum import IntEnum

class State(IntEnum):
    start = 1
    wall_spray_L = 2
    wall_nspray_L = 3
    wall_spray_R = 4
    wall_nspray_R = 5
    obj_turn_L = 6
    obj_turn_R = 7
    obj_turn_R_2 = 8
    obj_turn_L_2 = 9
    obj_turn_R_s = 11
    obj_turn_L_s = 12
    backup = 10


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3



########################################################################################
# Global variables
########################################################################################
    
rc = racecar_core.create_racecar()

#LIDAR Windows
FRONT_WINDOW = (-15, 15)
REAR_WINDOW = (170 , 190)
LEFT_WINDOW = ( 260, 280)
RIGHT_WINDOW = (80, 100)
LEFT_FRONT = (525, 719)
RIGHT_FRONT = (0, 195)
RIGHT_FRONT_WINDOW = (20, 40)
RIGHT_FRONT_WINDOW2 = (50, 70)
LEFT_FRONT_WINDOW = (290, 310)
LEFT_FRONT_WINDOW2 = (320, 340)

speed = 0
angle = 0
back = False

prevangle = 0
counter = 0
cur_state = State.obj_turn_R
counter = 0


BLUE = ((90, 255, 255), (120, 255, 255))  # The HSV range for the color blue
RED = ((0, 255, 255), (10, 255, 255))
GREEN = ((50, 255, 255), (70, 255, 255))


PURPLE = ((130,255,255), (140,255,255))
ORANGE = ((10,255,255), (20,255,255))


#CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_RIGHT = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_FLOOR = ((160, 0), (410, rc.camera.get_width()))

MIN_CONTOUR_AREA = 30

MAX_SPEED = 0.8

# When an object in front of the car is closer than this (in cm), start braking
BRAKE_DISTANCE = 150

# When a wall is within this distance (in cm), focus solely on not hitting that wall
PANIC_DISTANCE = 30

# When a wall is greater than this distance (in cm) away, exit panic mode
END_PANIC_DISTANCE = 32

# Speed to travel in panic mode
PANIC_SPEED = 0.3

# The minimum and maximum angles to consider when measuring closest side distance
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 60

# The angles of the two distance measurements used to estimate the angle of the left
# and right walls
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110

# When the front and back distance measurements of a wall differ by more than this
# amount (in cm), assume that the hallway turns and begin turning
TURN_THRESHOLD = 20

# The angle of measurements to average when taking an average distance measurement
WINDOW_ANGLE = 12


# Speeds
MAX_ALIGN_SPEED = 0.8
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 0.4

# Times
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2

# Cone finding parameters
MIN_CONTOUR_AREA = 100
MAX_DISTANCE = 250
REVERSE_DISTANCE = 50
STOP_REVERSE_DISTANCE = 60

CLOSE_DISTANCE = 30
FAR_DISTANCE = 120


back = False
#SLALOMING VARIABLES 
cur_mode = Mode.no_cones
counter = 0
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0
'''
RED = ((0, 100, 100), (10, 100, 100))
# HSV (0, 100, 100)
# BGR(255, 0, 0)
BLUE = ((100, 100, 100), (110, 100, 100))
# HSV (105, 100, 100)
# BGR (255, 127, 0)
GREEN = ((50, 100, 100), (65, 100, 100))
# HSV (60, 100, 100)
# BGR (0, 255, 0)
ORANGE = ((10, 100, 100), (20, 100, 100))
# HSV (15, 100, 100)
# BGR (0, 127, 255)
PURPLE = ((130, 100, 100), (140, 100, 100))
# HSV (135, 100, 100)
# BGR (255, 0, 127)
'''

########################################################################################
# Functions
########################################################################################
def start():

    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle
    global cur_state
    global back
    global prevangle
    global cur_line_state
    global counter
    global counter2
    global count1
    global prev_state
    count1 = 0
    counter = 0
    counter2 = 0
    #setting the state, whether the car "follows" the right or left wall
    cur_state = State.wall_spray_L
    #cur_state = State.obj_turn_R
    prev_state = State.wall_spray_L

    back = False
    # Have the car begin at a stop
    rc.drive.stop()
    prevangle = 0

    #Set global variables
    speed = 0
    angle = 0
    
    
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """


    ########################################################################################
    # Global Variables
    ########################################################################################

    global speed
    global angle
    global back
    global cur_state
    global prevangle
    global cur_line_state
    global counter 
    global counter2
    global count1
    global prev_state
    #getting image of surroundings for object detection and navigation
    image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
    depth_image = rc_utils.crop(depth_image, CROP_FLOOR[0], CROP_FLOOR[1])
    #print(str(CROP_FLOOR[1]))


    scan = rc.lidar.get_samples()
    speed = 0
    angle = 0
    count1 = 0
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    #speed = rt - lt
    if rc.controller.is_down(rc.controller.Button.A):
        cur_state = State.obj_turn_R
    if rc.controller.is_down(rc.controller.Button.B):
        cur_state = State.wall_spray_L
   
    # Use the left joystick to control the angle of the front wheels
    #angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    front_dist = rc_utils.get_lidar_closest_point(
        scan, (350, 10)
        )
    accel = rc.physics.get_linear_acceleration()
    print(str(accel[1]) + "acce")
    if (cur_state != State.backup and (front_dist[1])<20):
        print ("oops")
        prev_state = cur_state
        cur_state = State.backup 
        counter += rc.get_delta_time()
    '''
    if (cur_state != State.backup and accel[1]< 0):
        print ("oops")
        prev_state = cur_state
        cur_state = State.backup 
        counter += rc.get_delta_time() 
    '''   
    #rc.drive.set_speed_angle(speed, angle)
    if counter < 2.75 and counter > 0 and cur_state == State.backup:
        speed = -0.5
        angle = -0.4
        print(str(counter) + "time")
        counter += rc.get_delta_time()
        rc.drive.set_speed_angle(speed, angle)
    elif(cur_state == State.backup):
        print("done backup")
        cur_state = prev_state 
        cur_state = State.wall_spray_L
        counter = 0
    #the initial start
    if cur_state == State.start:
        #counter2 += rc.get_delta_time()
        isright = decideSide(scan)
        if (isright == True):
            cur_state = State.wall_nspray_R
        else:
            cur_state = State.wall_nspray_L
        
    if cur_state == State.wall_spray_L:
        #priority = getArTagColor(image)        
        print("line following state left")
        '''
        name = checkImage(image)
        if (name == "table idk"):
            cur_state = State.obj_turn_R
            turn_R()            
            #turn right slightly in relation to object (similar to cone turn, use same code)
            break
        else:
            wall_follow_L()
        '''
        #rc.display.show_color_image(image)
        #checks and identifies image using function 
        object = checkImage(image)
        front_dist = rc_utils.get_lidar_closest_point(
        scan, (350, 10)
        )
        """
        if (object == "dining_table" and front_dist < 50):
            cur_state = State.obj_turn_R
        """
        wall_follow_L(scan)
    if cur_state == State.wall_spray_R:
        #priority = getArTagColor(image)        
        print("line following state right")
        #name = checkImage(image)
        rc_utils.get_closest_point(depth_image)

        if (name == "table idk"):
            cur_state = State.obj_turn_L
            turn_L()            
            #turn right slightly in relation to object (similar to cone turn, use same code)
            #break
        else:
            wall_follow_R(scan)
    if cur_state == State.backup:
            counter += rc.get_delta_time()
            #print("counter right:")
            #print(counter)
            prev_state = cur_state
                #angle = rc_utils.clamp(angle, -1, 1)

           
        # Display the image to the screen
        #rc.display.show_color_image(image)
    
    if cur_state == State.obj_turn_R:
        front_dist = rc_utils.get_lidar_closest_point(
        scan, (350, 10)
        )
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (0,90))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 0, 90, 1, 0, True)
        print(closest_angle)
        rc.drive.set_speed_angle(speed, angle)
        if (closest_angle > 85):
            cur_state = State.obj_turn_R_s
    if cur_state == State.obj_turn_R_s:
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (270,360))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 360, 270, 1, -1, True)
        print(closest_angle)
        rc.drive.set_speed_angle(speed, angle)
        if (275 > closest_angle):
            cur_state = State.obj_turn_R_2
    if cur_state == State.obj_turn_R_2:
        print("switched")
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (180,270))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 270, 170, -1, 0, True)
        rc.drive.set_speed_angle(speed, angle)
        front_dist = rc_utils.get_lidar_closest_point(
        scan, (0, 20)
        )
        if (front_dist[1] < 50):
            cur_state = State.obj_turn_L
    if cur_state == State.obj_turn_L:
        front_dist = rc_utils.get_lidar_closest_point(
        scan, (350, 10)
        )
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (0,90))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 0, 90, -1, 0, True)
        print(closest_angle)
        rc.drive.set_speed_angle(speed, angle)
        if (closest_angle < 5):
            cur_state = State.obj_turn_L_s
    if cur_state == State.obj_turn_L_s:
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (90,180))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 90, 180, -1, 1, True)
        print(closest_angle)
        rc.drive.set_speed_angle(speed, angle)
        if (275 > closest_angle):
            cur_state = State.obj_turn_L_2
    if cur_state == State.obj_turn_L_2:
        print("switched")
        speed = 0.2
        closest_dist = rc_utils.get_lidar_closest_point(scan, (180,270))
        closest_angle = closest_dist[0]
        angle = rc_utils.remap_range(closest_angle, 270, 170, -1, 0, True)
        rc.drive.set_speed_angle(speed, angle)
        front_dist = rc_utils.get_lidar_closest_point(
        scan, (0, 20)
        )
        cur_state = State.wall_spray_L
        if (front_dist[1] < 50):
            cur_state = State.obj_turn_L
    if rc.controller.is_down(rc.controller.Button.X):
        rc.drive.set_speed_angle(1, 1)

     
############### OBJECT SCANNING FUNCTIONS ###################
#this function takes the image captured by the camera and inputs it into the ResNet 
#neural network, which classifies the image as either being an object or wall
def checkImage(image):
    newsize = (224, 224) 
    img = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    model = ResNet50(weights='imagenet')
    preds = model.predict(input_arr)
    print('Predicted:', decode_predictions(preds, top=1)[0])
    top_pred = decode_predictions(preds, top=1)[0]
    print(top_pred)
    prim_pred = top_pred[0]
    return prim_pred[1]

############### WALL FOLLOWING FUNCTIONS ###################
#Get furthest distance in front
def decideSide(scan):
    distance = rc_utils.get_lidar_average_distance()
#this function does wall following on the left side. It uses differences in distance
#to determine the turn angle. Further information on that algorithm is on the slides
def wall_follow_L(scan):

    global count1
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    left_front_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dif = left_front_dist - left_back_dist
    value = (-left_dif) + (60 - left_dist)
    angle = rc_utils.remap_range(
        value, -20, 20, -1, 1, True
    )
    print (str(left_front_dist) + " " + str(left_back_dist))
    print (str(left_dif) + " " + str(left_dist) + " " + str(angle))
    
    speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, MAX_SPEED, True)
    if (front_dist < 20):
        print(str(front_dist) + " front dist")
        count1 += rc.get_delta_time()
        if (count1 < 2): 
            rc.drive.set_speed_angle(-1, angle)
        else:
            count1 = 0
    # Choose speed based on the distance of the object in front of the car

    rc.drive.set_speed_angle(speed, angle)


def getFurthestDistanceAngle(scan):
    furthest_l = 0
    angle_l = 0
    for a in range(LEFT_FRONT[0], LEFT_FRONT[1]):
        val = scan[a]
        if a > furthest_l:
            furthest_l = val
            angle_l = a/2
    furthest_r = 0
    angle_r = 0 
    for b in range(RIGHT_FRONT[0], RIGHT_FRONT[1]):
        val = scan[b]
        if b > furthest_r:
            furthest_r = val
            angle_r = b/2
    furthest = 0
    angle = 0
    if furthest_l > furthest_r:
        furthest = furthest_l
        angle = angle_l
    else:
        furthest = furthest_r
        angle = angle_r
    return furthest, angle


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()