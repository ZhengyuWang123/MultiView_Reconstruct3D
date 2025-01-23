import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt

class Image_loader():
    def __init__(self, img_dir:str, downscale_factor:float):
        # loading the Camera intrinsic parameters K
        with open(img_dir + '\\K.txt') as f:
            # 内参转换到numpy float矩阵
            self.K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().split('\n')))))
            self.image_list = []
        # Loading the set of images
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)
        
        self.path = os.getcwd()
        self.factor = downscale_factor
        # 对图像做了下采样，图像尺寸变小，则焦距也要按同样比例变小，主点坐标也要缩放
        self.downscale()

    
    def downscale(self) -> None:
        '''
        Downscales the Image intrinsic parameter acc to the downscale factor
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor
    
    def downscale_image(self, image):
        # factor = 2 循环一次，factor = 4 循环两次
        for _ in range(1,int(self.factor / 2) + 1):
            # 每执行一次，就会将图像的宽高各缩小一半
            image = cv2.pyrDown(image)
        return image

class Sfm():
    def __init__(self, img_dir:str, downscale_factor:float = 2.0) -> None:
        '''
            Initialise and Sfm object.
        '''
        self.img_obj = Image_loader(img_dir,downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud 
        '''
        # point_2d_1 第一幅图像中匹配到的 2D 特征点
        # point_2d_2 第二幅图像中匹配到的同一批 2D 特征点，与 point_2d_1 一一对应。
        # projection_matrix_1.T 第一台相机的投影矩阵（3×4），在代码里做了 .T 转置传入。
        # 第二台相机的投影矩阵（3×4），与第一台类似，同样做了转置后传入
        # 一个完整的投影矩阵P是 (3×4)，可以将 3D 点(X,Y,Z,1)投影到 2D 图像上的(x,y,1)
        # 它会根据两幅图像里同一批空间点的投影坐标，以及两台相机的投影矩阵，求解出这些空间点的齐次坐标(4×N)
        # 2D(多张图片通过投影矩阵)-> 3D(齐次坐标)->非齐次坐标（3D点）
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        # (pt_cloud / pt_cloud[3])  齐次坐标=》（非齐次坐标）三角化后的 3D 点
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])    
    
    def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        #通过已知的3D点和其在2D图像中的投影点来估算摄像机的姿态（位置和方向）。
        # obj_point：3D 对应点坐标
        # image_point：2D 图像点坐标 它与 obj_point 一一对应

        if initial == 1:
            # (N,1,3) =》(N,3)
            obj_point = obj_point[:, 0 ,:]
            # OpenCV 的 PnP 接口一般要求输入的 2D 点是(N,2)
            image_point = image_point.T
            # 为了后面的一致性而做的转置
            rot_vector = rot_vector.T 
        # 用 RANSAC 的方式解 PnP 问题，即在众多 3D-2D 对应点中
        # 识别并排除异常值，然后求解最优的旋转和平移
        # rot_vector_calc：旋转向量（Rodrigues 形式）
        # tran_vector：平移向量 (3×1)
        # inlier内点索引(M,1) 数组 表示在 RANSAC 过程中被认为是“符合”模型的 3D-2D 匹配点索引。
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        # 如果传入的是 3×1 的旋转向量，就得到 3×3 的旋转矩阵
        # Converts a rotation matrix to a rotation vector or vice versa
        # rot_matrix 是 (3×3)，可视为相机外参中的旋转部分
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        # rot_matrix：(3×3) 的旋转矩阵，相机外参的 R。
        # tran_vector：(3×1) 的平移向量，相机外参的 t。
        # image_point & obj_point：过滤了内点后的一致 2D-3D 对应点
        # rot_vector：这里保留了滤完内点后的旋转向量
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector
    
    #reprojection_error 更像是独立地计算“当前估计好/假设好的外参和 3D 点下，投影误差是多少”，给你一个直观的数值做评估
    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        # 是在已知3D点、相机位姿和相机内参的条件下，
        # 把3D点重新投影到图像平面，和对应的真实2D观测点做对比，
        # 衡量它们之间的平均距离（即重投影误差）
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        # 把有些地方三角化没有转换为非齐次坐标的坐标从齐次坐标转化
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        # 3D => 2D
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        # 这里确保(N,2)
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points
    #外部用了least_squares反复调用，对 x0 做迭代更新以最小化重投影误差。
    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        calculates of the reprojection error during bundle adjustment
        returns error 
        '''
        # 它会根据打包在一起的各项（外参、内参、2D 点、3D 点）
        # 来计算重投影误差。
        # 然后 least_squares 等优化器会反复调用它来迭代更新参数。
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        # 40% 2D点
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        # 60% 3D点
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        # 旋转矩阵=》旋转向量
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        # 3D=》2D
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        # 平方误差
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        #旨在最小化重投影误差
        #优化（修正）相机外参、内参，以及 3D/2D 点的坐标。
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))
        # least_squares是一个非线性最小二乘问题求解器。
        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        # 函数返回 3D 点，2D 点，(3×4) 相机外参
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        # 形状是(N,6)，其中前 3 列是位置，后 3 列是颜色。
        verts = np.hstack([out_points, out_colors])


        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        # 计算每个点到质心的欧几里得距离 dist
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        # 去掉非常离群的点
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')


    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3 
        '''
        #试图找出在 image_points_1 和 image_points_2 中相同的 2D 点坐标，
        # 然后相应地在 image_points_3 做一些掩码处理
        # 它返回 4 个内容，以便后续在三角化或 PnP 等操作中使用。
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            # 近似距离”判断更好 ，== 可能会有精度隐患
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])
        # 对 image_points_2 和 image_points_3 做掩码并压缩
        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2


    def find_features(self, image_0, image_1) -> tuple:
        '''
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        '''
        # 创建 SIFT 检测器
        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)
        # Brute-Force Matcher，暴力匹配器，会计算两个描述子集合之间的距离
        bf = cv2.BFMatcher()
        # 对 desc_0 中每个描述子，在 desc_1 中找到距离最近和次近的两个描述子
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        #最近匹配和次近匹配差不多可能是杂质， 丢弃，需要m.distance 明显小于次近匹配 n.distance
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)
        # 输入两幅图。输出返回两个(N,2) 形状的 NumPy 数组，表示两幅图中互相匹配的特征点坐标
        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])

    def __call__(self, enable_bundle_adjustment:boolean=False):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
        # 投影矩阵
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4)) 
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        feature_0, feature_1 = self.find_features(image_0, image_1)

        # Essential matrix
        #可以在给定内参与匹配点对的情况下，通过 RANSAC 或其他方法稳健地估计出这个本质矩阵 𝐸
        # 并返回一个内点掩码以区分外点/内点。
        # 这里本质矩阵根据K的平移和旋转
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        #存在一定的**“尺度不确定”和“平移方向歧义”**(最多有四种可能解)。
        # 因此，recoverPose 的目的就是从 E 中选出一个“合理的 R,t”
        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]       
        # 我们要将上一帧的位姿与这一次的相对运动“累乘”起来
        # 若上一帧姿态是 {R0,t0}，本次相对运动是 {R,t}
        # R1​=RR0
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        #​ t1​=t0​+R0T​t
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())
        # 生成投影矩阵 P=K[R∣t]
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)
        # 投影矩阵三角化 得到一批 3D 点 points_3d
        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        # 再把这些 3D 点投影回来，求和真实测量的 feature_1 做比较，得到重投影误差 error
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
        #ideally error < 1
        print("REPROJECTION ERROR: ", error)
        # 再用 PnP 做进一步的位姿/外点筛选修正，获得更可靠的外参或点云。
        # 通过已知的3D点和其在2D图像中的投影点来估算摄像机的姿态（位置和方向）
        # 这里只是获得一些稳定的3D点
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2 
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.5
        for i in tqdm(range(total_images)):
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
            # 计算重投影误差并三角化 3D 点和 2D 点做投影比对，打印误差
            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity = 0)
        
            # 还用 pose_1, pose_2 对 cm_mask_0, cm_mask_1 再做一次三角化，以引入更多 3D 点。
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            # takes a long time to run
            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Bundle Adjusted error: ",error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector)) 
   


            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

if __name__ == '__main__':
    sfm = Sfm("Datasets\\entry-P10")
    sfm()

