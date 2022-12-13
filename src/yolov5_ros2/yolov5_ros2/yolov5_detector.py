import sys
import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Int16
from cv_bridge import CvBridge, CvBridgeError
from rclpy.utilities import remove_ros_args
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import TransformBroadcaster
from rclpy.callback_groups import ReentrantCallbackGroup

###自作ノード
from yolov5_ros2.detector import Detector, parse_opt



class ObjectDetection(Node):
    def __init__(self, **args):
        super().__init__('tomato_detection')
        self.detector = Detector(**args)
        self.bridge = CvBridge()
        
        # self.declare_parameter("camera_device", 0) #カメラデバイスどれ使うか(パラメータの宣言)
        # self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value #カメラ番号の取得
        self.running=False
        
        #tf
        self.broadcaster = TransformBroadcaster(self)
        self.frame_id = 'target'

        #publisher
        # self.pub = self.create_publisher(Int16, "chatter")


        #sSubscriber
        self.callback_group = ReentrantCallbackGroup()   # コールバックの並行処理のため
        self.sub_info = Subscriber(self, CameraInfo, 'camera/aligned_depth_to_color/camera_info',callback_group=self.callback_group)
        self.sub_color = Subscriber(self, Image, 'camera/color/image_raw',callback_group=self.callback_group)
        self.sub_depth = Subscriber(self, Image, 'camera/aligned_depth_to_color/image_raw',callback_group=self.callback_group)
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 10, 0.1) #同期処理　[]内のトピックが同期して処理される
        self.ts.registerCallback(self.images_callback) #ApproximateTimeSynchronizerで登録したtopicすべてのコールバックを行う



    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8') #ros→opencv
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]: #サイズ確認
            self.get_logger().warn('カラーと深度の画像サイズが異なる')
            return

        if self.running:
            img_color, result = self.detector.detect(img_color) #検出
            cv2.imshow('color', img_color)   #結果表示

            if len(result)!=0:
                count=0
                for r in result:
                    #bounding_boxの座標から対象物体の深度の代表値を探す
                    u1 = round(r.u1)
                    u2 = round(r.u2)
                    v1 = round(r.v1)
                    v2 = round(r.v2)
                    depth = np.median(img_depth[v1:v2+1, u1:u2+1])
                    
                    #bounding_boxの中心座標
                    u = round((r.u1 + r.u2) / 2)
                    v = round((r.v1 + r.v2) / 2)
                    
                    #カメラ座標系→現実座標系
                    """
                    fx,fy 内部パラメーター
                    (fx=fkx fx=fky  f:焦点距離　kx,ky:長さ→画素への変換パラメータ) 
                    cx,cy:カメラ座標の中心画素
                    z:物体距離
                    """
                    if depth != 0:
                        z = depth * 1e-3 #[mm→m変換]
                        fx = msg_info.k[0]
                        fy = msg_info.k[4]
                        cx = msg_info.k[2]
                        cy = msg_info.k[5]
                        x = z / fx * (u - cx)
                        y = z / fy * (v - cy)
                        self.get_logger().info(f'{r.name} ({x:.3f}, {y:.3f}, {z:.3f})') #三次元距離
                        #ts変換
                        ts = TransformStamped()
                        ts.header = msg_depth.header
                        ts.child_frame_id = self.frame_id+f"{count}"
                        ts.transform.translation.x = x
                        ts.transform.translation.y = y
                        ts.transform.translation.z = z
                        self.broadcaster.sendTransform(ts)
                        count+=1



def main():
    rclpy.init()
    opt = parse_opt(remove_ros_args(args=sys.argv))
    node = ObjectDetection(**vars(opt))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+Cが入力されました")  
        print("プログラム終了")  
    rclpy.shutdown()
