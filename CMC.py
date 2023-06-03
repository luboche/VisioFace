import cv2
import numpy as np
import os

def align_images_mask(gray_1, gray_2):
    # 计算相同部分的掩码
    diff_mask = cv2.compare(gray_1, gray_2, cv2.CMP_EQ)

    # 应用掩码
    gray_1_masked = cv2.bitwise_and(gray_1, gray_1, mask=diff_mask)
    gray_2_masked = cv2.bitwise_and(gray_2, gray_2, mask=diff_mask)

    # 对齐参数
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 对齐图像
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    (cc, warp_matrix) = cv2.findTransformECC(gray_1_masked, gray_2_masked, warp_matrix, warp_mode, criteria)

    # 对齐图像
    aligned_image = cv2.warpAffine(gray_2, warp_matrix, (gray_2.shape[1], gray_2.shape[0]),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_image, warp_matrix


def align_images_mask_top(image_1, image_2, if_mask=True):
    if image_1 != []:
        image_1 = image_1.astype(np.float32)  # 转换为32位浮点数
        image_2 = image_2.astype(np.float32)

        # 转换为灰度图像
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        if if_mask:
            # 计算掩码
            height, width = gray_1.shape
            mask = np.ones_like(gray_1, dtype=np.uint8)
            mask[:int(0.2 * height), :] = 0

            gray_1_masked = cv2.bitwise_and(gray_1, gray_1, mask=mask)
            gray_2_masked = cv2.bitwise_and(gray_2, gray_2, mask=mask)
        else:
            gray_1_masked = gray_1
            gray_2_masked = gray_2

        # 对齐参数
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # 对齐图像
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        (cc, warp_matrix) = cv2.findTransformECC(gray_1_masked, gray_2_masked, warp_matrix, warp_mode, criteria)

        # 对齐图像
        aligned_image = cv2.warpAffine(gray_2, warp_matrix, (gray_2.shape[1], gray_2.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        aligned_image = []

    return aligned_image, warp_matrix


def align_images_feature(gray_1, gray_2, mask_top=True, visualization=False):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    gray_1_ = cv2.convertScaleAbs(gray_1)
    gray_2_ = cv2.convertScaleAbs(gray_2)
    if mask_top:
        # 计算掩码
        height, width = gray_1.shape
        mask = np.ones_like(gray_1, dtype=np.uint8)
        mask[:int(0.2 * height), :] = 0

        gray_1_ = cv2.bitwise_and(gray_1_, gray_1_, mask=mask)
        gray_2_ = cv2.bitwise_and(gray_2_, gray_2_, mask=mask)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_1_, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_2_, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # # Draw top matches
    if visualization:
        imMatches = cv2.drawMatches(gray_1_, keypoints1, gray_2_, keypoints2, matches, None)
        cv2.imwrite("/data1/jing_li/otherfiles/YOLOV8/Anti_UAV/CMC_try/result/feature_mask_top/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    if not np.all(mask == 0):
        height, width = gray_1.shape
        aligned_image = cv2.warpPerspective(gray_2, h, (width, height))
    else:
        aligned_image, h = align_images_mask_top(gray_1, gray_2)

    return aligned_image, h


##########################################
#             图像对齐算法                #
##########################################
def align_images(gray_1, gray_2, mask_or_crop='crop', ECC_or_feature='ECC'):
    if gray_1 != []:  # 首张不为空
        if ECC_or_feature == 'ECC':
            gray_1 = gray_1.astype(np.float32)  # 转换为32位浮点数
            gray_2 = gray_2.astype(np.float32)
            # 转换为灰度图像
            gray_1 = cv2.cvtColor(gray_1, cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(gray_2, cv2.COLOR_BGR2GRAY)
            height = gray_1.shape[0]
            if mask_or_crop == 'crop':
                mask = np.ones_like(gray_1, dtype=np.uint8)
                mask[:int(0.2 * height), :] = 0
                gray_1_ = cv2.bitwise_and(gray_1, gray_1, mask=mask)
                gray_2_ = cv2.bitwise_and(gray_2, gray_2, mask=mask)
            else:
                # 计算截取的高度
                crop_height = int(height * 0.2)
                # 截取图片的上20%部分
                gray_1_ = gray_1[crop_height:, :]
                gray_2_ = gray_2[crop_height:, :]
            # 对齐参数
            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            # 对齐图像
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
            (cc, warp_matrix) = cv2.findTransformECC(gray_1_, gray_2_, warp_matrix, warp_mode, criteria)

            # 对齐图像
            aligned_image = cv2.warpAffine(gray_2, warp_matrix, (gray_2.shape[1], gray_2.shape[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            return aligned_image, warp_matrix

        elif ECC_or_feature == 'feature':
            MAX_FEATURES = 500
            GOOD_MATCH_PERCENT = 0.15
            gray_1_ = cv2.convertScaleAbs(gray_1)
            gray_2_ = cv2.convertScaleAbs(gray_2)
            if mask_or_crop == 'mask':
                # 计算掩码
                height = gray_1.shape[0]
                mask = np.ones_like(gray_1, dtype=np.uint8)
                mask[:int(0.2 * height), :] = 0

                gray_1_ = cv2.bitwise_and(gray_1_, gray_1_, mask=mask)
                gray_2_ = cv2.bitwise_and(gray_2_, gray_2_, mask=mask)
            elif mask_or_crop == 'crop':
                height = gray_1.shape[0]
                # 计算截取的高度
                crop_height = int(height * 0.2)
                # 截取图片的上20%部分
                gray_1_ = gray_1_[crop_height:, :]
                gray_2_ = gray_2_[crop_height:, :]

            # Detect ORB features and compute descriptors.
            orb = cv2.ORB_create(MAX_FEATURES)
            keypoints1, descriptors1 = orb.detectAndCompute(gray_1_, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray_2_, None)

            # Match features.
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)

            # Sort matches by score
            matches = list(matches)
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]

            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

            # 计算仿射矩阵(与单应矩阵不同)
            affine_matrix, _ = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC, ransacReprojThreshold=3)
            # 使用仿射矩阵将图像对齐
            aligned_image = cv2.warpAffine(gray_2, affine_matrix, (gray_1.shape[1], gray_1.shape[0]))

            return aligned_image, affine_matrix
    else:  # 首张为空
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        aligned_image = []
        return aligned_image, warp_matrix


def warp_pos(pos, warp_matrix):
    # 其次坐标
    p1 = np.array([pos[0, 0], pos[0, 1], 1])
    p2 = np.array([pos[0, 2], pos[0, 3], 1])

    p1_n = warp_matrix.dot(p1.T)
    p2_n = warp_matrix.dot(p2.T)

    return np.concatenate((p1_n, p2_n), axis=0)

# # 图像文件夹路径和输出文件夹路径
# image_folder = "/data1/jing_li/otherfiles/YOLOV8/Anti_UAV/CMC_try/pair"
# output_folder = "/data1/jing_li/otherfiles/YOLOV8/Anti_UAV/CMC_try/result/feature_mask_top"

# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)

# # 对每组图像对进行对齐并拼接显示
# for i in range(7):  # 图像对的子文件夹编号范围为0到6
#     image_path_1 = os.path.join(image_folder, str(i), "0.jpg")
#     image_path_2 = os.path.join(image_folder, str(i), "1.jpg")

#     # 加载图像
#     image_1 = cv2.imread(image_path_1)
#     image_1 = image_1.astype(np.float32)  # 转换为32位浮点数

#     image_2 = cv2.imread(image_path_2)
#     image_2 = image_2.astype(np.float32)

#     # 转换为灰度图像
#     gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
#     gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

#     # 对齐图像
#     # aligned_image,warp_matrix = align_images(gray_1, gray_2)
#     aligned_image,warp_matrix = align_images_mask(gray_1, gray_2)
#     # aligned_image,warp_matrix = align_images_feature(gray_1, gray_2)
#     print('pair {},warp_matrix:{}'.format(i,warp_matrix))

#     # 创建透明度融合图像
#     alpha = 0.5  # 设置融合的透明度权重
#     blended_image_l = cv2.addWeighted(gray_1, 1-alpha, gray_2, alpha, 0)
#     blended_image_r = cv2.addWeighted(gray_1, 1-alpha, aligned_image, alpha, 0)
#     concatenated_image = np.concatenate((blended_image_l, blended_image_r), axis=1)

#     # 保存拼接后的图像
#     output_path = os.path.join(output_folder, f"concatenated_{i}.jpg")
#     cv2.imwrite(output_path, concatenated_image)