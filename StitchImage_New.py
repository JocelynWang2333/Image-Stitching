import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    top, bot, left, right = 300, 300, 0, 1200

    img1 = cv.imread('1-1.jpg',1)
    img2 = cv.imread('1-2.jpg',1)

    # 扩充src的边缘，将图像变大，然后以各种外插方式自动填充图像边界
    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)

    # 使用sift算法对图片特征进行提取
    sift = cv.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    # 其中kp1和kp2保存了特征点的位置和大小，des保存了n维向量的描述子
    print('=======>Detecting key points!')
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)

    # FLANN parameters
    # 使用快速最近邻搜索包匹配器对特征点进行匹配， 随机k_d树算法
    print('=======>Matching key points!')
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # 指定递归次数

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)

            pts2.append(kp2[m.trainIdx].pt) # 此匹配对应的训练(模板)图像的特征描述子索引
            pts1.append(kp1[m.queryIdx].pt) # 此匹配对应的查询图像的特征描述子索引
            # then queryIdx 指 keypoints1 和 trainIdx 指 keypoints2
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3)
    plt.show()

    rows, cols = srcImg.shape[:2]
    # 取彩色图片的高、宽两项参数（对应列表的多少行/多少列）
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # 源平面中点的坐标矩阵
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 目标平面中点的坐标矩阵

        # 找到两个平面之间的转换矩阵
        print('=======>Random sampling and computing the homography matrix!')
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv.WARP_INVERSE_MAP)


        for col in range(0, cols):
            # 找出左图中的
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break

        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        # 设置黑色画布值，初始化为零矩阵
        res = np.zeros([rows, cols, 3], np.uint8)
        # 遍历图一
        for row in range(0, rows):
            for col in range(0, cols):
                # 若左图为空，则拼接图为经过投影变幻的图二
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                # 若右图为空，则拼接图为左边图一
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                # 若左右图皆为非空，则考虑将重叠部分数列合并时赋予权重进行操作
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        cv.imwrite('1-output.JPG', res)
        print('\n=======>Output saved!')
        # cv2.imshow('output.JPG', merge_image)
        # if cv2.waitKey() == 27:
        #     cv2.destroyAllWindows()
        # opencv is bgr, matplotlib is rgb
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # show the result
        plt.figure()
        plt.imshow(res)
        plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
