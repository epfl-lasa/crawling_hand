#!/usr/bin/env python
import sys

sys.path.append('..')
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
import tf
from std_msgs.msg import Header


# import tools.rotations as rot

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_scale = 0.001  # depth_scale

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo,
                                                self.camera_info_callback)

        self.result_image_pub = rospy.Publisher("/detection_image", Image, queue_size=10)
        self.poses_pub = rospy.Publisher("/object_poses", PoseArray, queue_size=10)
        self.hsv_debug_image_pub = rospy.Publisher("/hsv_debug_image", Image, queue_size=10)
        self.depth_image = None
        self.rgb_image = None

        # 上一次检测到的物体位姿
        self.last_poses = {
            'yellow': None,
            'blue': None,
            'red': None,

        }

        # 预期尺寸的面积范围（像素数，根据实际情况调整）
        self.min_area = 1000  # 最小面积
        self.max_area = 6000  # 最大面积

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.rect_ratio = 6

    def camera_info_callback(self, camera_info):
        self.camera_matrix = np.array(camera_info.K).reshape(3, 3)
        self.dist_coeffs = np.array(camera_info.D)
        # rospy.loginfo("Camera info received.")

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def image_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.depth_image is not None:
                self.detect_objects()
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_min_area_rect_points(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def detect_objects(self):
        hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)

        # 定义颜色范围
        colors = {
            'yellow': ((16, 103, 78), (25, 255, 255), [60, 30, 15]),
            'blue': ((90, 103, 78), (120, 255, 255), [30, 30, 30]),
            'red1': ((0, 103, 78), (15, 255, 255), [30, 30, 30]),  # 低红色范围
            'red2': ((135, 103, 78), (180, 255, 255), [30, 30, 30]),  # 高红色范围

        }

        # 创建一个空白的HSV分割结果图像
        hsv_debug_image = np.zeros_like(self.rgb_image)

        # 初始化新的位姿
        new_poses = {
            'yellow': None,
            'blue': None,
            'red': None
        }

        for color, (lower, upper, dimensions) in colors.items():
            if color == 'red1':
                mask1 = cv2.inRange(hsv_image, lower, upper)
                continue
            elif color == 'red2':
                mask2 = cv2.inRange(hsv_image, lower, upper)
                mask = cv2.bitwise_or(mask1, mask2)  # 合并两个红色掩码
                # continue
            else:
                mask = cv2.inRange(hsv_image, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hsv_debug_image[mask > 0] = self.rgb_image[mask > 0]  # 将分割结果复制到调试图像

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    box = self.get_min_area_rect_points(contour)
                    l1 = np.linalg.norm(box[1, :] - box[0, :])
                    l2 = np.linalg.norm(box[2, :] - box[1, :])

                    if l1 >10 and l2> 10 and 1/self.rect_ratio < l1/l2 < self.rect_ratio:
                        cv2.drawContours(self.rgb_image, [box], -1, (0, 255, 0), 2)

                        # 提取最小矩形包络的点云
                        mask_fill = np.zeros_like(mask)
                        cv2.fillPoly(mask_fill, [box], 255)
                        points_y, points_x = np.where(mask_fill == 255)
                        depths = self.depth_image[points_y, points_x] * self.depth_scale
                        valid_points = depths > 0

                        points_x = points_x[valid_points]
                        points_y = points_y[valid_points]
                        depths = depths[valid_points]

                        points_3d = self.deproject_pixels_to_points(points_x, points_y, depths)

                        if len(points_3d) >= 4:
                            points_3d = np.array(points_3d)
                            pose = self.estimate_pose(points_3d, color)
                            if pose is not None:
                                new_poses[color.split('1')[0].split('2')[0]] = pose

        # 发布检测到的位姿
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "camera_link"

        for color in ['yellow', 'blue', 'red']:
            if new_poses[color] is not None:
                self.last_poses[color] = new_poses[color]
            if self.last_poses[color] is not None:
                pose_array.poses.append(self.last_poses[color])
                self.publish_tf_transform(self.last_poses[color], color)

        self.poses_pub.publish(pose_array)
        self.publish_detection_image()
        self.publish_hsv_debug_image(hsv_debug_image)

    # def deproject_pixel_to_point(self, x, y, depth):
    #     cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
    #     fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
    #     point3d = [(x - cx) * depth / fx, (y - cy) * depth / fy, depth]
    #     return point3d
    def deproject_pixels_to_points(self, x, y, depth):
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        points_3d = np.zeros((len(x), 3))
        points_3d[:, 0] = (x - cx) * depth / fx
        points_3d[:, 1] = (y - cy) * depth / fy
        points_3d[:, 2] = depth
        return points_3d

    def publish_tf_transform(self, pose, color):
        frame_id = "camera_color_optical_frame"
        child_frame_id = f"{color}_frame"
        time = rospy.Time.now()

        self.tf_broadcaster.sendTransform(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            time,
            child_frame_id,
            frame_id
        )

    def estimate_pose(self, points_3d, color):
        # 计算质心
        centroid = np.mean(points_3d, axis=0)

        # 去质心
        points_centered = points_3d - centroid

        # 计算协方差矩阵
        cov_matrix = np.cov(points_centered.T)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 特征向量按特征值升序排列，所以长边方向是具有最大特征值的特征向量
        # 取特征值最大的特征向量为长边方向
        v1 = eigenvectors[:, np.argmax(eigenvalues)]

        # Step 5: 取最小特征值对应的特征向量为v3
        v3 = eigenvectors[:, np.argmin(eigenvalues)]

        # Step 6: 确保v3与相机z轴[0, 0, 1]夹角为钝角
        camera_z_axis = np.array([0, 0, 1])
        if np.dot(v3, camera_z_axis) < 0:
            v3 = -v3
        # Step 7: 计算新的y轴方向
        v2 = np.cross(v3, v1)

        # Step 8: 构建旋转矩阵
        R = np.column_stack((v1, v2, v3))
        # 通过特征向量构建旋转矩阵
        # rotation_matrix = eigenvectors

        # 转换为四元数
        quaternion = mat2quat(R)

        pose = Pose()
        pose.position.x = centroid[0]
        pose.position.y = centroid[1]
        pose.position.z = centroid[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        rospy.loginfo("Detected %s pose: %s", color, pose)
        return pose

    def publish_detection_image(self):
        try:
            detection_image_msg = self.bridge.cv2_to_imgmsg(self.rgb_image, "bgr8")
            detection_image_msg.header = Header()
            detection_image_msg.header.stamp = rospy.Time.now()
            self.result_image_pub.publish(detection_image_msg)
            # rospy.loginfo("Detection image published.")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def publish_hsv_debug_image(self, hsv_debug_image):
        try:
            hsv_debug_image_msg = self.bridge.cv2_to_imgmsg(hsv_debug_image, "bgr8")
            hsv_debug_image_msg.header = Header()
            hsv_debug_image_msg.header.stamp = rospy.Time.now()
            self.hsv_debug_image_pub.publish(hsv_debug_image_msg)
            # rospy.loginfo("HSV debug image published.")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
