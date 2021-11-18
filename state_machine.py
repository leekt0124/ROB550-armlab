"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2
import csv

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
        """
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        """
        self.waypoints = WaypointRecording()

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
        k_move = 2.5
        k_accel = 1.0/5
        min_move_time = 0.5
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints.arm_coords)):
            current_position = self.rxarm.get_positions() # Get current position
            next_position = self.waypoints.arm_coords[i] # Get next position
            difference = np.absolute(np.subtract(current_position, next_position)) # Difference between current and next position
            print('Difference Position')
            print(difference)
            max_angle_disp = np.amax(difference)# Find highest angle displacement
            print('Max Displacement')
            print(max_angle_disp)
            moving_time = k_move * max_angle_disp# Multiply the above by constant to get time
            if (moving_time < min_move_time):
                moving_time = min_move_time
            print('Moving Time')
            print(moving_time)
            accel_time = k_accel * moving_time
            print('Acceleration Time')
            print(accel_time)
            self.rxarm.set_moving_time(moving_time)# Do set moving time
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)
            if(self.waypoints.gripper_state[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(1.0)
        self.next_state = "idle"

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
                print(tag_position_c[id1,0])
        tag_position_c = np.transpose(tag_position_c).astype(np.float32)
        for i in range(4):
            tag_position_c[:, i] /=  tag_position_c[2,i]

        tag_position_i = np.dot(self.camera.intrinsic_matrix,tag_position_c).astype(np.float32)

        print("U V coordinates")
        print(tag_position_i)
        #tag_position_i = tag_position_i
        self.status_message = "Calibration - Completed Calibration"

        (success, rot_vec, trans_vec) = cv2.solvePnP(self.camera.tag_locations.astype(np.float32), np.transpose(tag_position_i[:2, :]).astype(np.float32), self.camera.intrinsic_matrix,self.camera.dist_coefficient, flags = cv2.SOLVEPNP_ITERATIVE)

        #print("translational",trans_vec)
        #print(success)

        dst = cv2.Rodrigues(rot_vec)
        dst = np.array(dst[0])
        # print(dst)
        trans_vec = np.squeeze(trans_vec)
        self.camera.extrinsic_matrix[:3, :3] = dst
        self.camera.extrinsic_matrix[:3, 3] = trans_vec

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
                print(type(self.waypoints.arm_coords))
                print(self.waypoints.arm_coords[i])
                print(self.waypoints.arm_coords[i][0])
    def load_waypoints(self):
        with open("teach_repeat.csv", "r") as file:
            reader = csv.reader(file, delimeter = '\t')
            for row in reader:
                print(row)

    def clear_waypoints(self):
        self.waypoints.arm_coords = []
        self.waypoints.gripper_state = []

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
