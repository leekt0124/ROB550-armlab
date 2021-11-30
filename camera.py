"""!
Class to represent the camera.
"""

import cv2
import math
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.BlocksDetectedFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        #self.intrinsic_matrix = np.array([[904.3, 0, 696.0], [0, 906.1, 361.9], [0, 0, 1]]) # Average intrinsic_matrix
        self.intrinsic_matrix = np.array([[908.3550415039062, 0, 642.5927124023438], [0, 908.4041137695312, 353.12652587890625], [0, 0, 1]]) # Factory intrinsic_matrix
        self.extrinsic_matrix = np.linalg.inv(np.array([[1, 0, 0, -20], [0, -1, 0, 180], [0, 0, -1, 973], [0, 0, 0, 1]]).astype(np.float32))
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.dist_coefficient = np.array([[0.17931543290615082, -0.5406785011291504, -0.0007807965739630163, -0.0004374352574814111, 0.4746035635471344]
])
        """
            ADDED ZERO TO THE END OF EACH POINT IN THE NP ARRAY TO MAKE IT A 4X3 MATRIX
        """
        self.tag_locations = np.array([[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]])
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBlocksDetectionFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.BlocksDetectedFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections

                    # use self.VideoFrame
                    # use self.DepthFrameRaw

        """

        '''
        BLOCK DETECTION
        '''

        # TODO: Find blocks higher than one stack
        # - probably need to do contours for each depth threshold

        # TODO: Tune depth and retune these parameters, depth misread is throwing off smaller blocks
        # TODO: Can't find small blocks in upper left corner
        # - If I turn up the floor, it creates a big contour in the lower right, making it useless
        lower = 905
        upper = 953

        # Image Frame
        self.BlocksDetectedFrame = self.VideoFrame
        # TODO: Add blur to create better contour detection? Try to use blur to smooth inside of block but leave outside alone
        rgb_image = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        cnt_image = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR)
        # Depth data in right format
        depth_data = self.DepthFrameRaw

        #cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
        """mask out arm & outside board"""
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (275,120),(1100,715), 255, cv2.FILLED)
        cv2.rectangle(mask, (575,414),(736,720), 0, cv2.FILLED)
        cv2.rectangle(self.BlocksDetectedFrame , (275,120),(1100,715), (255, 0, 0), 2)
        cv2.rectangle(self.BlocksDetectedFrame , (575,414),(736,720), (255, 0, 0), 2)
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Show thresh
        _, self.block_contours, self.cnt_hierarchies = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.BlocksDetectedFrame, self.block_contours, -1, (255, 0, 255), 3)
        #print(self.block_contours)
        num_contours = np.shape(self.block_contours)[0]
        print(num_contours)

        '''
        BLOCK LABELING
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        # TODO: Find which color is which under a better range
        # - Sample colors around the board
        # - Generate heat map of colors
        # - Find individual thresholds for colors

        # RGB Color List
        colors = list((
            {'id': 'red', 'color': (10, 10, 127)},
            {'id': 'orange', 'color': (30, 75, 150)},
            {'id': 'yellow', 'color': (30, 150, 200)},
            {'id': 'green', 'color': (20, 60, 20)},
            {'id': 'blue', 'color': (100, 50, 0)},
            {'id': 'violet', 'color': (100, 40, 80)})
        )

        # TODO: Populate this
        colors_hsv = list((
            {'id': 'red', 'color': (0, 170, 90)},
            {'id': 'red', 'color': (160, 170, 90)},
            {'id': 'orange', 'color': (11, 180, 180)},
            {'id': 'yellow', 'color': (25, 220, 204)},
            {'id': 'green', 'color': (70, 213, 61)},
            {'id': 'blue', 'color': (104, 203, 94)},
            {'id': 'violet', 'color': (115, 102, 95)})
        )

        def retrieve_area_color_rgb(data, contour, labels):
                mask = np.zeros(data.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean = cv2.mean(data, mask=mask)[:3]

                min_dist = (np.inf, None)
                for label in labels:
                    d = np.linalg.norm(label["color"] - np.array(mean))
                    if d < min_dist[0]:
                        min_dist = (d, label["id"])
                return min_dist[1]

        def retrieve_area_color_hsv(data, contour, labels):
            mask = np.zeros(data.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean = cv2.mean(data, mask=mask)[:3]
            print(mean)
#            for elt in mean:
#                if(elt > 180):
#                    elt = elt-180
            min_dist = (np.inf, None)
            for label in labels:
                #d = np.linalg.norm(label["color"] - np.array(mean))
                #print(label["color"][0])
                d = math.sqrt(abs(label["color"][0] * label["color"][0] - mean[0]*mean[0]))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            print("~~")
            return min_dist[1]

        def u_v_d_to_world(u,v,d):
            z = d

            uv_coords = np.array([float(u), float(v), float(1)])
            intrinsic_inv = np.linalg.inv(self.intrinsic_matrix)

            c_coords =  np.matmul(intrinsic_inv, uv_coords)
            c_coords *= z
            c_coords = np.append(c_coords, [float(1)], axis=0)
            w_coords = np.matmul(np.linalg.inv(self.extrinsic_matrix), c_coords)
            return w_coords[:3]

        def is_contour_bad(c):
            # Check size
            threshold_area = 400
            area = cv2.contourArea(contour)
            if(area) < threshold_area:
                return 1

            # Check if square


        self.block_detections = []
        #self.block_detections = np.zeros((num_contours,3))
        print(np.shape(self.block_detections))
        i = 0
        for contour in self.block_contours:
            print("RGB COLOR MATCHING")
            color = retrieve_area_color_rgb(rgb_image, contour, colors)
            print(color)
            print("HSV COLOR MATCHING")
            color = retrieve_area_color_hsv(hsv_image, contour, colors_hsv)
            print(color)
            print("~~~~~~~")
            theta = cv2.minAreaRect(contour)[2]
            # Classify block size
            # TODO: Tune threshold, apply depth into calculation too (smaller blocks look bigger when stacked high)
            threshold_area = 250
            area = cv2.contourArea(contour)
            print(area)
            # If block is too small (error in contour reading), skip
            if is_contour_bad(contour):
                # Skip over labelling and don't incnlude in block list
                continue

            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cz = self.DepthFrameRaw[cy][cx]


            cv2.putText(self.BlocksDetectedFrame , color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
            cv2.putText(self.BlocksDetectedFrame , str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
            cv2.putText(self.BlocksDetectedFrame , str(int(area)), (cx+50, cy), font, 1.0, (0,255,0), thickness=2)

            print(color, int(theta), cx, cy)
            self.block_detections.append(u_v_d_to_world(cx,cy,cz))
            i += 1

        print("Block Detections: ")
        print(self.block_detections)

        cv2.drawContours(self.BlocksDetectedFrame, self.block_contours, -1, (255, 0, 255), 3)
        #self.processVideoFrame()
        #cv2.rectangle(self.VideoFrame, (275,120),(1100,720), (255, 0, 0), 2)
        #cv2.rectangle(self.VideoFrame, (575,414),(723,720), (255, 0, 0), 2)




    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data


        #print(self.camera.tag_detections.detections[0].pose.pose.pose)
        #print('--------')
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blocks window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            blocks_frame = self.camera.convertQtBlocksDetectionFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, blocks_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    "Blocks window",
                    cv2.cvtColor(self.camera.BlocksDetectedFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
