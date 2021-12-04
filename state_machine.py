"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import timeit
import numpy as np
import rospy
import cv2
import csv
import math
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from kinematics import clamp

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

class Mask():
    def __init__(self, x_center, y_center, radius):
        self.x_center = x_center
        self.y_center = y_center
        self.radius = radius

class WaypointRecording():
    def __init__(self):
        self.arm_coords = []
        self.gripper_state = []

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.gripper_state = 1

        self.waypoints = WaypointRecording()
        # self.waypoints.arm_coords = [
        #     [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
        #     [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
        #     [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
        #     [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
        #     [0.0,             0.0,      0.0,         0.0,     0.0],
        #     [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
        #     [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
        #     [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
        #     [np.pi/2,         0.5,     0.3,      0.0,     0.0],
        #     [0.0,             0.0,     0.0,      0.0,     0.0]]

        # self.waypoints.gripper_state = [1,1,1,1,1,1,1,1,1,1]



    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        #self.rxarm.set_moving_time(2.0)
        k_move = 1.2
        k_accel = 1.0/10
        min_move_time = 0.5
        self.status_message = "State: Execute - Executing motion plan"
        # current_position = self.rxarm.get_positions()
        # print(current_position)
        # print(type(current_position))
        for i in range(len(self.waypoints.arm_coords)):
            current_position = self.rxarm.get_positions()# Get current position
            current_position = [clamp(deg) for deg in current_position]
            next_position = self.waypoints.arm_coords[i] # Get next position
            next_position = [clamp(deg) for deg in next_position]
            # print("next_position = ", next_position)
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1])) # Difference between current and next position
            # print('Difference Position')
            # print(difference)
            max_angle_disp = np.amax(difference)# Find highest angle displacement
            # print('Max Displacement')
            # print(max_angle_disp)
            moving_time = k_move * max_angle_disp# Multiply the above by constant to get time
            # print("max_ang = ", max_angle_disp)
            # print("moving_time = ", moving_time)
#            if (moving_time < min_move_time):
#                moving_time = min_move_time
            #print('Moving Time')
            #print(moving_time)
            accel_time = k_accel * moving_time
            #print('Acceleration Time')
            #print(accel_time)
            self.rxarm.set_moving_time(moving_time)# Do set moving time
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)
            if(self.waypoints.gripper_state[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(0.1)
        self.next_state = "idle"
        self.clear_waypoints()

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        tag_position_c = np.zeros((4,3))
        print('Calibration')
        if len(self.camera.tag_detections.detections) < 4:
            self.status_message = "Not enough tags"
        else:
            for i in range(4):
                id1 = self.camera.tag_detections.detections[i].id[0] - 1
                tag_position_c[id1,0] = self.camera.tag_detections.detections[i].pose.pose.pose.position.x
                tag_position_c[id1,1] = self.camera.tag_detections.detections[i].pose.pose.pose.position.y
                tag_position_c[id1,2] = self.camera.tag_detections.detections[i].pose.pose.pose.position.z

        tag_position_c = np.transpose(tag_position_c).astype(np.float32)
        tag_position_c_scaled = tag_position_c * 1000
        for i in range(4):
            tag_position_c[:, i] /=  tag_position_c[2,i]

        tag_position_i = np.dot(self.camera.intrinsic_matrix,tag_position_c).astype(np.float32)

        #print("U V coordinates")
        #print(tag_position_i)
        #tag_position_i = tag_position_i
        self.status_message = "Calibration - Completed Calibration"


        (success, rot_vec, trans_vec) = cv2.solvePnP(self.camera.tag_locations.astype(np.float32), np.transpose(tag_position_i[:2, :]).astype(np.float32), self.camera.intrinsic_matrix,self.camera.dist_coefficient, flags = cv2.SOLVEPNP_ITERATIVE)


        dst = cv2.Rodrigues(rot_vec)
        dst = np.array(dst[0])

        trans_vec = np.squeeze(trans_vec)
        self.camera.extrinsic_matrix[:3, :3] = dst
        self.camera.extrinsic_matrix[:3, 3] = trans_vec


        # Tilted plane fix
        uv_coords = tag_position_i.astype(int)
        intrinsic_inv = np.linalg.inv(self.camera.intrinsic_matrix)
        c_coords =  np.matmul(intrinsic_inv, uv_coords)

        for i in range(4):
            tag_position_c[:, i] /=  tag_position_c[2,i]
            z = self.camera.DepthFrameRaw[uv_coords[1,i]][uv_coords[0,i]]
            c_coords[:,i] *= z

        #print(c_coords.shape)
        #print([float(1),float(1),float(1),float(1)])
        c_coords = np.append(c_coords, [[float(1),float(1),float(1),float(1)]], axis=0)
        w_coords = np.matmul(np.linalg.inv(self.camera.extrinsic_matrix), c_coords)
        #print(w_coords)

        # Cross product of tilted frame
        id1 = 1;
        id2 = 2;
        cross_w = np.cross(w_coords[:3,id1], w_coords[:3,id2])
        b=np.linalg.norm(cross_w)
        cross_w = cross_w / b

        w_points = np.append(np.expand_dims(w_coords[:3,id1], axis = 1),np.expand_dims(w_coords[:3,id2], axis = 1), axis = 1)
        w_points = np.append(w_points,np.expand_dims(cross_w, axis = 1), axis = 1)
        #print(w_points)

        # Cross product of true locations
        true_locations = np.transpose(self.camera.tag_locations)
        cross_t = np.cross(true_locations[:,id1], true_locations[:,id2])
        t=np.linalg.norm(cross_t)
        cross_t = cross_t / t

        t_points = np.append(np.expand_dims(true_locations[:,id1], axis = 1),np.expand_dims(true_locations[:,id2], axis = 1), axis = 1)
        t_points = np.append(t_points,np.expand_dims(cross_t, axis = 1), axis = 1)
        #print(t_points)

        # Cross product for rotation axis
        rot_axis = np.cross(cross_w, cross_t)
        mag_rot=np.linalg.norm(rot_axis)
        rot_axis = rot_axis / mag_rot
        #print(rot_axis)

        # Angle of rotation
        dot_product = np.dot(cross_w, cross_t)
        angle = -np.arccos(dot_product)/2

        # Quaternion rotation around the axis
        q_rotation = Quaternion(axis = rot_axis, angle = angle)
        #print(q_rotation)

        # From Quaternion to rotation
        #R = q_rotation.transformation_matrix
        #print(q_rotation.transformation_matrix)
        R = np.array([[1, 0, 0, 0],[0 ,math.cos(angle), - math.sin(angle), 0],[0, math.sin(angle), math.cos(angle), 10], [0, 0, 0, 1]])


        self.camera.extrinsic_matrix = np.matmul(self.camera.extrinsic_matrix, np.linalg.inv(R))
        #
        #print(angle)
        #print(R)
        self.camera.processDepthFrame()


        print(self.camera.extrinsic_matrix)


    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

    def add_waypoint(self):
        self.waypoints.arm_coords.append(self.rxarm.get_positions())
        self.waypoints.gripper_state.append(self.gripper_state)
        print(self.waypoints.arm_coords)

    def add_gripper(self, state):
        self.gripper_state = state

    def save_waypoints(self):
        # Write out to file
        with open("teach_repeat.csv", "wb") as file:
            writer = csv.writer(file)
            for i in range(len(self.waypoints.arm_coords)):
                writer.writerow(self.waypoints.arm_coords[i][0])
                writer.writerow(self.waypoints.gripper_state[i][0])
                # print(type(self.waypoints.arm_coords))
                # print(self.waypoints.arm_coords[i])
                # print(self.waypoints.arm_coords[i][0])
    def load_waypoints(self):
        with open("teach_repeat.csv", "r") as file:
            reader = csv.reader(file, delimeter = '\t')
            for row in reader:
                print(row)

    def clear_waypoints(self):
        self.waypoints.arm_coords = []
        self.waypoints.gripper_state = []

    def pick_click(self):
        print('Camera Coordinates')
        print(self.camera.last_click)
        x = self.camera.last_click[0]
        y = self.camera.last_click[1]
        z = self.camera.DepthFrameRaw[y][x]
        # from click get world coordinates
        w_coords = self.camera.u_v_d_to_world(x,y,z)

        self.pick("big",w_coords)

    def pick(self, block_size, w_coords, block_theta=0):
        # Append phi angle to w_coords

        phi_i = 175
        phi = phi_i
        w_coords = np.append(w_coords, phi)

        # Increase z by 30 mm (avoid hitting ground)
        w_coords_up = w_coords.copy()
        w_coords_down = w_coords.copy()

        w_coords_up[2] += 200.0

        if w_coords_down[2] < 5.0:
            w_coords_down[2] += 20.0

        # # w_coords.append(np.pi - 0.02)
        # print('World Coordinates up')
        # print(w_coords_up)
        # print('World Coordinates down')
        # print(w_coords_down)

        # IK
        # block_rot = 0;

        while ((any(np.isnan(self.rxarm.world_to_joint(w_coords_up)[0])) or any(np.isnan(self.rxarm.world_to_joint(w_coords_down)[0]))) and phi >= 80):
#            print("trying phi = ", phi)
            phi -= 5
            w_coords[3] = phi

            # Increase z by 30 mm (avoid hitting ground)
            w_coords_up = w_coords.copy()
            w_coords_down = w_coords.copy()
            w_coords_up[2] += 80.0

            if w_coords_down[2] < 5.0:
                w_coords_down[2] += 20.0



        if block_size == "small":
                w_coords_down[2] += 5


        joint_angles_up = self.rxarm.world_to_joint(w_coords_up)

        joint_angles_up.flatten()
        if phi == phi_i:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up[0][0])))
        else:
            block_rot = 0
        # if block_rot < 0:
        #     block_rot += 180
        joint_angles_up = np.append(joint_angles_up, block_rot)
        self.waypoints.arm_coords.append(joint_angles_up)
        self.waypoints.gripper_state.append(1)
        # print(self.waypoints.arm_coords)

        joint_angles_down = self.rxarm.world_to_joint(w_coords_down)
        joint_angles_down.flatten()
        joint_angles_down = np.append(joint_angles_down, block_rot)

        self.waypoints.arm_coords.append(joint_angles_down)
        self.waypoints.gripper_state.append(0)

        self.waypoints.arm_coords.append(joint_angles_up)
        self.waypoints.gripper_state.append(0)

        self.execute()
        # print(self.waypoints.arm_coords)

        # define trajectory based on click
        # use inverse kinematics to calculate joint position
        # Add to join positions waypoint to self.waypoints
        # make sure end effector closes at the end


    def place_click(self):
        print('Camera Coordinates')
        print(self.camera.last_click)
        x = self.camera.last_click[0]
        y = self.camera.last_click[1]
        z = self.camera.DepthFrameRaw[y][x]
        # from click get world coordinates
        c_coords = [x,y,z]
        w_coords = self.camera.u_v_d_to_world(x,y,z)
        self.place(c_coords, w_coords)

    def place(self, c_coords, w_coords, block_theta=0, mask_placement=0, block_size="big"):
        # Append phi angle to w_coords
        phi_i = 175
        phi = phi_i
        w_coords = np.append(w_coords, phi)

        # Increase z by 30 mm (avoid hitting ground)
        w_coords_up = w_coords.copy()
        w_coords_down = w_coords.copy()
        w_coords_up[2] += 200.0
        w_coords_down[2] += 30.0

        # w_coords.append(np.pi - 0.02)
        # print('World Coordinates up')
        # print(w_coords_up)
        # print('World Coordinates down')
        # print(w_coords_down)

        # IK

        while ((any(np.isnan(self.rxarm.world_to_joint(w_coords_up)[0])) or any(np.isnan(self.rxarm.world_to_joint(w_coords_down)[0]))) and phi >= 80):
            # print("trying phi = ", phi)
            phi -= 5
            w_coords[3] = phi

            # Increase z by 30 mm (avoid hitting ground)
            w_coords_up = w_coords.copy()
            w_coords_down = w_coords.copy()
            w_coords_up[2] += 80.0

            if w_coords_down[2] < 5.0:
                w_coords_down[2] += 20.0


        # TODO: Not this, manually adjust z for bad calibration in stack
        if(w_coords_down[0] > 315):
            w_coords_down[2] = w_coords_down[2] + 10

        joint_angles_up = self.rxarm.world_to_joint(w_coords_up)
        # print("first = ", joint_angles_up)

        joint_angles_up.flatten()
        # print("second = ", joint_angles_up)

        if phi == phi_i:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up[0][0])))
        else:
            block_rot = 0
        # if block_rot < 0:
        #     block_rot += 180
        # print("block_rot = ", block_rot)
        joint_angles_up = np.append(joint_angles_up, block_rot)

        self.waypoints.arm_coords.append(joint_angles_up)
        self.waypoints.gripper_state.append(0)
        # print(self.waypoints.arm_coords)

        joint_angles_down = self.rxarm.world_to_joint(w_coords_down)

        joint_angles_down.flatten()
        joint_angles_down = np.append(joint_angles_down, block_rot)
        self.waypoints.arm_coords.append(joint_angles_down)
        self.waypoints.gripper_state.append(1)
        # same as pick but open end effector at the end

        self.waypoints.arm_coords.append(joint_angles_up)
        self.waypoints.gripper_state.append(1)

        # Create mask for placement
        # TODO: Convert world back to camera coords
        if(mask_placement):
            print("PLACING MASK !!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if(w_coords[0] < 0):
                disp = 1.4
            else:
                disp = 1.4
            if(block_size == "small"):
                msk = Mask(int(c_coords[0]-disp), int(c_coords[1]), 50)
            else:
                msk = Mask(int(c_coords[0]-disp), int(c_coords[1]), 100)
            self.camera.mask_list.append(msk)

        self.execute()

    def pick_sort(self):

        # Image coordinates
        destination_right_uv = [[805,485,973],[805,545,973],[805, 585,973],[865,485,973],[865,545,973],[865, 585,973],[925,485,973],[925,545,973],[925, 585,973]]
        destination_left_uv = [[515,485,973],[515,545,973],[515, 585,973],[455,485,973],[455,545,973],[455, 585,973],[395,485,973],[395,545,973],[395, 585,973]]

        destination_right = [list(self.camera.u_v_d_to_world(dest[0], dest[1], dest[2])) for dest in destination_right_uv]
        destination_left = [list(self.camera.u_v_d_to_world(dest[0], dest[1], dest[2])) for dest in destination_left_uv]

        print(destination_right_uv)
        print(destination_right)
        print(destination_left_uv)
        print(destination_left)
        #destination_right = [[155,50,0],[155,-30,0],[205,-110,0]]
        #destination_left = [[-155,50,0],[-155,-30,0],[-205,-110,0]]

        self.camera.blockDetector()

        i = 0
        j = 0

        while len(self.camera.block_detections) > 0:
            for block in self.camera.block_detections:
                w_coords = block.coord
                print(w_coords)
                self.pick(block.size, w_coords, block.theta)
                if(block.size == "big"):
                    self.place(destination_right_uv[i], destination_right[i], 0, 1, block.size)
                    i += 1
                else:
                    self.place(destination_left_uv[j], destination_left[j], 0, 1, block.size)
                    j += 1
            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(1)

        self.camera.mask_list = []

    # Event 2
    def pick_stack(self):

        # Image coordinates
        destination_bases_uv = [[875,600,968],[975,510,963],[825,460,968]]
        destination_buff_uv = [[515,485,968],[515,545,968],[515, 585,968],[455,485,968],[455,545,968],[455, 585,968],[395,485,968],[395,545,968],[395, 585,968]]

        destination_bases = [list(self.camera.u_v_d_to_world(dest[0], dest[1], dest[2])) for dest in destination_bases_uv]
        destination_buff = [list(self.camera.u_v_d_to_world(dest[0], dest[1], dest[2])) for dest in destination_buff_uv]

        self.camera.blockDetector()
        i = 0
        j = 0
        state = "stack_bigs"
        block_detect_process = []
        while len(self.camera.block_detections) > 0:
            # Sort
            for block in self.camera.block_detections:
                if(block.size == "big"):
                    block_detect_process.append(block)
            for block in self.camera.block_detections:
                if(block.size == "small"):
                    block_detect_process.append(block)

            # Change state
            if(block_detect_process[0].size == "big"):
                state = "stack_bigs"
            print("HELLO !!!!!!!!!!!!!!!!!!!!")
            print(block_detect_process[0].coord[2])
            print(block_detect_process[0].size)
            if(block_detect_process[0].coord[2] < 40 and block_detect_process[0].size == "small"):
                print(block_detect_process[0].coord[2])
                print(block_detect_process[0].size)
                state = "stack_smalls"

            if(state == "stack_bigs"):
                for block in block_detect_process:
                    w_coords = block.coord
                    print(w_coords)
                    # Grab all big blocks first to form the stack bases

                    if(block.size == "big"):
                        self.pick(block.size, w_coords, block.theta)
                        self.place(destination_bases_uv[i%3], destination_bases[i%3], 0, 1, block.size)
                        print("PLACING AT DESTINATION: ")
                        print(destination_bases[i%3])
                        destination_bases[i%3][2] = destination_bases[i%3][2] + 50
                        print("NEW DESTINATION: ")
                        print(destination_bases[i%3])
                        i += 1
                    elif(block.size == "small" and block.coord[1] > 120):
                        self.pick(block.size, w_coords, block.theta)
                        self.place(destination_buff_uv[j], destination_buff[j], 0, 0, block.size)
                        j += 1

            elif(state == "stack_smalls"):
                print("I AM IN STACK SMALL STATE")
                for block in block_detect_process:
                    w_coords = block.coord
                    print(w_coords)

                    self.pick(block.size, w_coords, block.theta)
                    self.place(destination_bases_uv[i%3], destination_bases[i%3], 0, 1, block.size)
                    destination_bases[i%3][2] += 25
                    i += 1
            block_detect_process = []


            self.rxarm.sleep()
            rospy.sleep(2)
            #
            self.camera.blockDetector(660, 1200, 400, 700)
            rospy.sleep(1)
        print("I AM DONE, DELETING ALL MASKS !!!!!")
        print("Uastohuaontshuntsoeahurcagpuoasneuhoaeust")
        self.camera.mask_list = []

        # Event 3
        def line_up(self):
            pass

        # Event 4
        def stack_high(self):
            pass

        # Event 5
        def to_sky(self):
            pass

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)
