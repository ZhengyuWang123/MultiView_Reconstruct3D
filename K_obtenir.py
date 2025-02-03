import cv2
import numpy as np
import glob

# 1. 准备棋盘格参数
# 这里假设棋盘有 7×9 个内角点（注意这指的是可检测到的角点数，而非方格数）
CHECKERBOARD = (7, 9)

# 物理世界中，每个方格的尺寸（单位自定，如 mm），如果不关心绝对尺度，可设为 1.0
square_size = 20.0  

# 2. 准备储存角点的列表
# 3D 点在世界坐标系中的坐标，例如 (0,0,0), (1,0,0)... 这里做成数组
# 这相当于一张平面的坐标，因为棋盘在同一平面上。
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# 让 [x, y] 依次填充棋盘平面的坐标，比如 (x格子, y格子, 0)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 如果你想加上物理尺寸，则再乘以一个方格尺寸
objp = objp * square_size

# 用于存储所有图像的 3D 点、2D 点
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像坐标系中的二维点

# 3. 读取标定图像
images = glob.glob(r"F:\github\MultiView_Reconstruct3D\calib_images\*.jpg")
print("Images found:", images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. 检测棋盘角点
    # criteria 用于亚像素精确化迭代，下面设置迭代 30 次或精度达到 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 如果找到角点，则进一步亚像素精确化，存入 objpoints 和 imgpoints
    if ret == True:
        objpoints.append(objp)

        # 亚像素精确化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (可选) 在图像上显示检测到的角点，用于可视化检查
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 5. 进行标定
# 传入收集到的 3D 点和对应的 2D 点，计算相机内参、畸变系数等
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 6. 打印结果
print("标定结果是否成功（越接近1说明越成功）:", ret)
print("相机内参矩阵 (cameraMatrix):\n", cameraMatrix)
print("畸变系数 (distCoeffs):\n", distCoeffs)
# rvecs, tvecs 是每张标定图像对应的旋转、平移向量，可按需查看

# 7. 计算平均重投影误差，用于评估标定质量
total_error = 0
for i in range(len(objpoints)):
    # 将 3D 点投影回图像，再和实际角点对比
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print("平均重投影误差: ", mean_error)

# 8. 可将 cameraMatrix, distCoeffs 保存到文件（如 .txt、.yaml、.npz 等）
np.savez("camera_calib_result.npz",
         cameraMatrix=cameraMatrix,
         distCoeffs=distCoeffs,
         mean_error=mean_error)
