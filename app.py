import base64
import time
from flask import Flask, render_template, request, Response, stream_with_context
from flask import abort, redirect, url_for
import argparse
import numpy as np
import cv2
import math
import json
from math import *
import os
import torch
import torchvision
from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces

# 创建应用程序
videodata = []


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points_6 = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ], dtype=float) / 4.5

        self.model_points_14 = np.array([
            (6.825897, 6.760612, 4.402142),
            (1.330353, 7.122144, 6.903745),
            (-1.330353, 7.122144, 6.903745),
            (-6.825897, 6.760612, 4.402142),
            (5.311432, 5.485328, 3.987654),
            (1.789930, 5.393625, 4.413414),
            (-1.789930, 5.393625, 4.413414),
            (-5.311432, 5.485328, 3.987654),
            (2.005628, 1.409845, 6.165652),
            (-2.005628, 1.409845, 6.165652),
            (2.774015, -2.080775, 5.048531),
            (-2.774015, -2.080775, 5.048531),
            (0.000000, -3.116408, 6.097667),
            (0.000000, -7.415691, 4.070434)], dtype=float)

        self.model_points_68 = np.array([
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638, 7.929818, -44.84258],
            [-66.850058, 26.07428, -43.141114],
            [-59.790187, 42.56439, -38.635298],
            [-48.368973, 56.48108, -30.750622],
            [-34.121101, 67.246992, -18.456453],
            [-17.875411, 75.056892, -3.609035],
            [0.098749, 77.061286, 0.881698],
            [17.477031, 74.758448, -5.181201],
            [32.648966, 66.929021, -19.176563],
            [46.372358, 56.311389, -30.77057],
            [57.34348, 42.419126, -37.628629],
            [64.388482, 25.45588, -40.886309],
            [68.212038, 6.990805, -42.281449],
            [70.486405, -11.666193, -44.142567],
            [71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795, -7.268147],
            [-37.8048, -61.996155, -0.442051],
            [-24.022754, -61.033399, 6.606501],
            [-11.635713, -56.686759, 11.967398],
            [12.056636, -57.391033, 12.051204],
            [25.106256, -61.902186, 7.315098],
            [38.338588, -62.777713, 1.022953],
            [51.191007, -59.302347, -5.349435],
            [60.053851, -50.190255, -11.615746],
            [0.65394, -42.19379, 13.380835],
            [0.804809, -30.993721, 21.150853],
            [0.992204, -19.944596, 29.284036],
            [1.226783, -8.414541, 36.94806],
            [-14.772472, 2.598255, 20.132003],
            [-7.180239, 4.751589, 23.536684],
            [0.55592, 6.5629, 25.944448],
            [8.272499, 4.661005, 23.695741],
            [15.214351, 2.643046, 20.858157],
            [-46.04729, -37.471411, -7.037989],
            [-37.674688, -42.73051, -3.021217],
            [-27.883856, -42.711517, -1.353629],
            [-19.648268, -36.754742, 0.111088],
            [-28.272965, -35.134493, 0.147273],
            [-38.082418, -34.919043, -1.476612],
            [19.265868, -37.032306, 0.665746],
            [27.894191, -43.342445, -0.24766],
            [37.437529, -43.110822, -1.696435],
            [45.170805, -38.086515, -4.894163],
            [38.196454, -35.532024, -0.282961],
            [28.764989, -35.484289, 1.172675],
            [-28.916267, 28.612716, 2.24031],
            [-17.533194, 22.172187, 15.934335],
            [-6.68459, 19.029051, 22.611355],
            [0.381001, 20.721118, 23.748437],
            [8.375443, 19.03546, 22.721995],
            [18.876618, 22.394109, 15.610679],
            [28.794412, 28.079924, 3.217393],
            [19.057574, 36.298248, 14.987997],
            [8.956375, 39.634575, 22.554245],
            [0.381549, 40.395647, 23.591626],
            [-7.428895, 39.836405, 22.406106],
            [-18.160634, 36.677899, 15.121907],
            [-24.37749, 28.677771, 4.785684],
            [-6.897633, 25.475976, 20.893742],
            [0.340663, 26.014269, 22.220479],
            [8.444722, 25.326198, 21.02552],
            [24.474473, 28.323008, 5.712776],
            [8.449166, 30.596216, 20.671489],
            [0.205322, 31.408738, 21.90367],
            [-7.198266, 30.844876, 20.328022]])

        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    def get_euler_angle(self, rotation_vector):
        # calc rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transform to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x ** 2 + y ** 2)
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y ** 2 + z ** 2)
        roll = math.atan2(t3, t4)

        return pitch, yaw, roll

    def solve_pose_by_6_points(self, image_points):
        """
    Solve pose from image points
    Return (rotation_vector, translation_vector) as pose.
    """
        points_6 = np.float32([
            image_points[30], image_points[36], image_points[45],
            image_points[48], image_points[54], image_points[8]])

        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_6,
            points_6,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return rotation_vector, translation_vector

    def solve_pose_by_14_points(self, image_points):
        points_14 = np.float32([
            image_points[17], image_points[21], image_points[22], image_points[26], image_points[36],
            image_points[39], image_points[42], image_points[45], image_points[31], image_points[35],
            image_points[48], image_points[54], image_points[57], image_points[8]])

        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_14,
            points_14,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return rotation_vector, translation_vector

    def solve_pose_by_68_points(self, image_points):
        judge, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_68,  # 68关键点三维坐标
            image_points,  # 68关键点 x，y坐标
            self.camera_matrix,  # 相机内参
            self.dist_coeefs,  # 相机内参
            rvec=self.r_vec,  # 旋转矩阵 起始搜素位置
            tvec=self.t_vec,  # 位移矩阵 起始搜索
            useExtrinsicGuess=True)  # 使用给定值搜索
        if judge:
            self.r_vec = rotation_vector
            self.t_vec = translation_vector
            return rotation_vector, translation_vector  # 分别返回solvePnP得到的旋转矩阵和位移矩阵
        else:
            return [0, 0, 0], 0




def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint.pth.tar",
                        type=str)
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
args = parse_args()
checkpoint = torch.load(args.model_path, map_location=device)
pfld_backbone = PFLDInference().to(device)
auxiliarynet = AuxiliaryNet().to(device)
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])
pfld_backbone.eval()
auxiliarynet.eval()
pfld_backbone = pfld_backbone.to(device)
auxiliarynet = auxiliarynet.to(device)


def mymodel(args, picture):
    time1 = time.perf_counter()
    print(1, time1)
    global point
    if picture.all() == None:
        return []
    data = []
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    img = picture
    height, width = img.shape[:2]
    bounding_boxes, landmarks = detect_faces(img)
    point = np.zeros((98, 3))
    pose_estimator = PoseEstimator(img_size=img.shape)
    time2 = (time.perf_counter())
    print(2, time2)
    for box in bounding_boxes:

        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2  # center of x
        cy = y1 + h // 2  # center of y
        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)
        time25 = (time.perf_counter())
        print(2.5, time25)
        cropped = img[y1:y2, x1:x2]  # 人脸剪切
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                         cv2.BORDER_CONSTANT, 0)
        time26 = (time.perf_counter())
        print(2.6, time26)
        input = cv2.resize(cropped, (112, 112))  # 固定大小
        input = transform(input).unsqueeze(0).to(device)
        features, landmarks = pfld_backbone(input)
        angle = auxiliarynet(features)[0]
        # print(angle)
        # print(angle)
        angle = angle.detach().numpy()

        time3 = (time.perf_counter())
        print(3, time3)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]

        for i, (x, y) in enumerate(pre_landmark.astype(np.int32)):
            point[i][0] = x
            point[i][1] = y
            # data.append([x, y])
        time4 = (time.perf_counter())
        print(4, time4)
        solvepoint = np.float32([
            point[0][:2],
            point[2][:2],
            point[4][:2],
            point[6][:2],
            point[8][:2],
            point[10][:2],
            point[12][:2],
            point[14][:2],
            point[16][:2],
            point[18][:2],
            point[20][:2],
            point[22][:2],
            point[24][:2],
            point[26][:2],
            point[28][:2],
            point[30][:2],
            point[32][:2],
            point[33][:2],
            point[34][:2],
            point[35][:2],
            point[36][:2],
            point[37][:2],
            point[42][:2],
            point[43][:2],
            point[44][:2],
            point[45][:2],
            point[46][:2],
            point[51][:2],
            point[52][:2],
            point[53][:2],
            point[54][:2],
            point[55][:2],
            point[56][:2],
            point[57][:2],
            point[58][:2],
            point[59][:2],
            point[60][:2],
            point[61][:2],
            point[63][:2],
            point[64][:2],
            point[65][:2],
            point[67][:2],
            point[68][:2],
            point[69][:2],
            point[71][:2],
            point[72][:2],
            point[73][:2],
            point[75][:2],
            point[76][:2],
            point[77][:2],
            point[78][:2],
            point[79][:2],
            point[80][:2],
            point[81][:2],
            point[82][:2],
            point[83][:2],
            point[84][:2],
            point[85][:2],
            point[86][:2],
            point[87][:2],
            point[88][:2],
            point[89][:2],
            point[90][:2],
            point[91][:2],
            point[92][:2],
            point[93][:2],
            point[94][:2],
            point[95][:2],
        ])
        pose = pose_estimator.solve_pose_by_68_points(solvepoint)
        # point[:, :2] = np.resize(data, (98, 2))
        time5 = (time.perf_counter())
        print(5, time5)
        z, y, x = pose[1]
        print(pose[1])
        z = z / 57.29
        y = y / 57.29
        x = x / 57.29
        rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, -1]])
        x1 = np.dot(np.dot(np.dot([point[0][0], point[0][1], 0], rx), ry), rz)
        x2 = np.dot(np.dot(np.dot([point[32][0], point[32][1], 0], rx), ry), rz)
        distance = math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 + (x1[2] - x2[2]) ** 2)
        z = np.zeros(98)
        # 脸颊到下巴  耳朵高度为0
        z[0] = 0
        z[1] = 0.02
        z[2] = 0.04
        z[3] = 0.06
        z[4] = 0.09
        z[5] = 0.13
        z[6] = 0.18
        z[7] = 0.23
        z[8] = 0.29
        z[9] = 0.36
        z[10] = 0.43
        z[11] = 0.49
        z[12] = 0.56
        z[13] = 0.63
        z[14] = 0.70
        z[15] = 0.73
        z[16] = 0.75
        for i in range(17, 33):
            z[i] = z[32 - i]
        # -----------------

        # 嘴巴  基准点 z[85] 下嘴唇中
        z[85] = z[16]
        z[76] = z[82] = z[88] = z[92] = z[85] - 0.03
        z[83] = z[87] = z[85] - 0.02
        z[84] = z[86] = z[85] - 0.01
        z[94] = z[90] = z[85] + 0.02
        z[89] = z[91] = z[95] = z[93] = z[85] + 0.01
        z[77] = z[81] = z[85] - 0.01
        z[78] = z[80] = z[85] + 0.03
        z[79] = z[85] + 0.02

        # 鼻子  基准点z[51] 鼻梁
        z[51] = z[15]
        z[52] = z[51] + 0.03
        z[53] = z[52] + 0.03
        z[54] = z[53] + 0.03
        z[57] = z[52]
        z[56] = z[58] = z[57] - 0.01
        z[55] = z[59] = z[57] - 0.02

        # 眼睛   基准点z[96] 眼球
        z[96] = z[97] = z[14]
        z[62] = z[70] = z[66] = z[74] = z[96] + 0.01
        z[61] = z[63] = z[69] = z[71] = z[96] + 0.05
        z[65] = z[67] = z[73] = z[75] = z[96] + 0.05
        z[60] = z[64] = z[68] = z[72] = z[96]

        # 眉毛   基准点 z[35] 眉毛中间
        z[35] = z[44] = z[15]
        z[40] = z[48] = z[35]
        z[34] = z[41] = z[45] = z[47] = z[35] - 0.02
        z[33] = z[46] = z[35] - 0.04
        z[37] = z[38] = z[42] = z[50] = z[35] - 0.02
        z[36] = z[39] = z[43] = z[49] = z[35] - 0.01
        time6 = (time.perf_counter())
        print(6, time6)
        for i in range(98):
            point[i][2] = z[i] * distance

        for i in range(98):
            point[i] = np.dot(np.dot(np.dot(point[i], rx), ry), rz)
        time7 = (time.perf_counter())
        print(7, time7)

    return point.tolist()


def get_video_data():
    global args
    if os.path.exists('rev_image.jpg'):
        pic = cv2.imread('rev_image.jpg')
        data = mymodel(args, pic)
        temp = [str((i1)) for i2 in data for i1 in i2]
        retemp = '*'.join(temp)
        return retemp
    return []


@app.errorhandler(404)  # 传入要处理的错误代码
def page_not_found(e):  # 接受异常对象作为参数
    return render_template('404.html'), 404  # 返回模板和状态码


@app.route('/progress')  #
def progress():
    @stream_with_context
    def generate():
        ratio = get_video_data()  #
        while 1:
            yield "data:" + ratio + "\n\n"
            ratio = get_video_data()
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/receiveImage/', methods=["POST"])
def receive_image():
    if request.method == "POST":
        data = request.data.decode('utf-8')
        json_data = json.loads(data)
        str_image = json_data.get("imgData")
        img = base64.b64decode(str_image)
        img_np = np.frombuffer(img, dtype='uint8')
        new_img_np = cv2.imdecode(img_np, 1)
        cv2.imwrite('rev_image.jpg', new_img_np)
    return Response('upload')


@app.route('/picture', methods=['POST', 'GET'])
def picture():
    global args
    if os.path.exists('aaa.jpg'):
        pic = cv2.imread('aaa.jpg')
        data = mymodel(args, pic)
        return render_template('picture.html', before=1, data=data)
    return render_template('picture.html', before=0, data=[])


@app.route('/getImg/', methods=['GET', 'POST'])
def getImg():
    imgData = request.files["image"]
    imgName = imgData.filename
    imgData.save('aaa.jpg')
    url = '/static/upload/img/' + imgName
    return redirect(url_for('picture'))


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        temp = request.form.get('choice')

        if temp == "picture":
            return redirect(url_for('picture'))
        elif temp == "video":
            return redirect(url_for('video'))
    return render_template('main.html')


@app.route('/')
def index():
    return redirect(url_for('main'))  # 重定向到


if __name__ == "__main__":
    app.run(debug=True)  # 启动应用程序，不
