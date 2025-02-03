import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt

class Image_loader():
    def __init__(self, img_dir:str, downscale_factor:float):
        #  Chargement des paramètres intrinsèques de la caméra K
        with open(img_dir + '\\K.txt') as f:
            # Conversion de la matrice intrinsèque en matrice NumPy de float
            self.K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().split('\n')))))
            self.image_list = []
        # Chargement de l'ensemble d'images
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)
        
        self.path = os.getcwd()
        self.factor = downscale_factor
        # Un sous-échantillonnage a été effectué sur les images, ce qui réduit leur taille.
        # Par conséquent, la distance focale doit être réduite de la même proportion
        # et les coordonnées du point principal doivent également être mises à l'échelle.
        #self.downscale()

    
    def downscale(self) -> None:
        '''
        Réduit les paramètres intrinsèques de l'image en fonction du facteur de réduction.
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor
    
    def downscale_image(self, image):
        # Si factor = 2, la boucle s'exécute une fois; si factor = 4, la boucle s'exécute deux fois.
        for _ in range(1,int(self.factor / 2) + 1):
            # À chaque itération, la largeur et la hauteur de l'image sont réduites de moitié.
            image = cv2.pyrDown(image)
        return image

class FeatureExtractor():

    def find_features(self, image_0, image_1) -> tuple:
        '''
        Détection de caractéristiques à l'aide de l'algorithme SIFT et KNN.
        Renvoie les points-clés (caractéristiques) de image_0 et image_1.
        '''
        # Création du détecteur SIFT
        #sift = cv2.xfeatures2d.SIFT_create()
        self.sift = cv2.SIFT_create()

        key_points_0, desc_0 = self.sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = self.sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)
        # Brute-Force Matcher, un appariement exhaustif qui calcule la distance entre deux ensembles de descripteurs.
        bf = cv2.BFMatcher()
        # Pour chaque descripteur dans desc_0, trouve les deux descripteurs les plus proches dans desc_1.
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        #Si la distance entre le meilleur et le deuxième meilleur appariement est trop proche,
        #c'est probablement du bruit. On l'écarte, nous avons besoin de m.distance nettement inférieure à n.distance
        #(test du ratio de Lowe).
        
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)
        
        #feature = [m for (m, n) in matches]
        # En entrée, deux images. En sortie, deux tableaux NumPy de forme (N,2) 
        # représentant les coordonnées des points correspondants dans chaque image.

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])
        
class Triangulator():
    def triangulation(self, projection_matrix_1, projection_matrix_2, point_2d_1, point_2d_2) -> tuple:
        '''
        Fait la triangulation de points 3D à partir de vecteurs 2D et de matrices de projection.
        Renvoie la matrice de projection de la première caméra, celle de la deuxième caméra et le nuage de points.
        '''
        # point_2d_1 : les points caractéristiques 2D extraits de la première image
        # point_2d_2 : les mêmes points caractéristiques 2D correspondants dans la deuxième image, un-à-un avec point_2d_1.
        # projection_matrix_1.T : la matrice de projection (3×4) de la première caméra, transposée avant d'être transmise.
        # La matrice de projection (3×4) de la deuxième caméra, similaire à la première, transposée également.
        # Une matrice de projection complète P est (3×4), elle projette un point 3D (X,Y,Z,1) sur un point 2D (x,y,1).
        # À partir des coordonnées projetées dans les deux images pour les mêmes points 3D, ainsi que des matrices
        # de projection des deux caméras, elle calcule les coordonnées homogènes (4×N) de ces points 3D.
        # 2D (plusieurs images via la matrice de projection) -> 3D (coordonnées homogènes) -> coordonnées non homogènes (points 3D).
        # pt_cloud(4,N)

        pts_2d_1 = point_2d_1.T  # (2, N)
        pts_2d_2 = point_2d_2.T  # (2, N)

        pt_cloud = cv2.triangulatePoints(projection_matrix_1,projection_matrix_2,pts_2d_1, pts_2d_2)
        # (pt_cloud / pt_cloud[3]) : coordonnées homogènes -> (coordonnées non homogènes) des points 3D triangulés.
        pt_cloud_4N = pt_cloud / pt_cloud[3]
        points_4d = pt_cloud_4N.T

        mask_valid = (points_4d[:, 2] > 0) & (points_4d[:, 2] < 1e4)
        points_4d_filtered = points_4d[mask_valid]  # (M,4)
        pts_2d_1_filtered = pts_2d_1[:, mask_valid]  # (2, M)
        pts_2d_2_filtered = pts_2d_2[:, mask_valid]

        points_4d_filtered_4N = points_4d_filtered.T

        #return point_2d_1.T, point_2d_2.T, (pt_cloud / pt_cloud[3])    
        return pts_2d_1_filtered, pts_2d_2_filtered, points_4d_filtered_4N

class PnPSolver():
    def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
        '''
        Détermine la pose d'un objet à partir de correspondances 3D-2D en utilisant RANSAC.
        Renvoie la matrice de rotation, la matrice de translation, les points d'image,
        les points d'objet et le vecteur de rotation.
        '''
        # Estime la pose (position et orientation) de la caméra en utilisant les points 3D connus
        # et leurs projections 2D correspondantes.
        # obj_point : coordonnées 3D correspondantes
        # image_point : coordonnées 2D dans l'image, correspondantes un-à-un avec obj_point.
        # Utilise RANSAC pour résoudre le problème de PnP, c'est-à-dire parmi de nombreuses correspondances 3D-2D.
        # Identifie et élimine les valeurs aberrantes, puis résout la meilleure rotation et translation.
        # rot_vector_calc : vecteur de rotation (forme de Rodrigues).
        # tran_vector : vecteur de translation (3×1).
        # inlier : indices (M,1) indiquant les correspondances 3D-2D jugées cohérentes par le modèle RANSAC.
        # Si on transmet un vecteur de rotation (3×1), on obtient une matrice de rotation (3×3).
        # Convertit une matrice de rotation en vecteur de rotation ou inversement.
        # rot_matrix est (3×3), elle peut être considérée comme la partie rotation de la pose caméra.
        # tran_vector : vecteur de translation (3×1), la composante t de la pose caméra.
        # image_point & obj_point : les correspondances 2D-3D cohérentes après le filtrage des inliers.
        # rot_vector : le vecteur de rotation conservé après filtrage des inliers.

        if initial == 1:
            # (N,1,3) =》(N,3)
            obj_point = obj_point[:, 0 ,:]
            image_point = image_point.T
            rot_vector = rot_vector.T 
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

class ReprojectionErrorCalculator():
    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        '''
        Calcule l'erreur de reprojection, c'est-à-dire la distance entre les points projetés et les points réels.
        Renvoie l'erreur totale et les points d'objet.
         '''
        # Avec les points 3D, la pose de la caméra et les paramètres intrinsèques connus,
        # on reprojette les points 3D dans le plan de l'image et on les compare aux points 2D observés,
        # pour mesurer la distance moyenne entre eux (c'est l'erreur de reprojection).
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        # Convertit, si nécessaire, certains points encore au format homogène en coordonnées non homogènes
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        # 3D => 2D
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        # Assure ici le format (N,2)
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points
    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        Calcule l'erreur de reprojection pendant le bundle adjustment.
        Retourne l'erreur.
        '''
        # La fonction least_squares l'appelle de façon itérative sur x0 afin de minimiser l'erreur de reprojection.
        # Elle se base sur les divers paramètres regroupés (pose, intrinsecs, points 2D, points 3D)
        # pour calculer l'erreur de reprojection.
        # Puis la fonction least_squares ou un autre solveur l'appelle de manière itérative pour mettre à jour les paramètres.

        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        # 40% = points 2D
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        # 60% = points 3D
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        # Matrice de rotation -> vecteur de rotation
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        # 3D -> 2D
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        # Erreur au carré
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

class BundleAdjuster():
    def __init__(self, reprojection_error_calculator: ReprojectionErrorCalculator):
        # Nécessite un calculateur d'erreur de reprojection.
        self.reproj_calc = reprojection_error_calculator
    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        # Visant à minimiser l'erreur de reprojection
        # Optimise (corrige) les paramètres externes de la caméra, les paramètres internes,
        # ainsi que les coordonnées des points 3D/2D.

        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))
        values_corrected = least_squares(self.reproj_calc.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        # La fonction renvoie les points 3D, les points 2D et la matrice externe (3×4) de la caméra.

        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

class PlySaver():
    def to_ply(self, path, point_cloud, colors, image_list) -> None:
        '''
        Génère un fichier .ply permettant d'ouvrir le nuage de points.
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        # Calcule la distance euclidienne de chaque point au centre de gravité.
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        # Exclut les points trop éloignés (outliers).
        indx = np.where(dist < np.mean(dist) + 300)
        #threshold = np.mean(dist) * 3
        #indx = np.where(dist < threshold)

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
        name_part = image_list[0].split('\\')[-2]
        out_file = path + '\\res\\' + name_part + '.ply'
        with open(out_file, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

class CommonPointsFinder():
    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        Trouve les points communs entre image 1 et 2, et entre image 2 et 3.
        Renvoie les points communs de image 1-2, de image 2-3, ainsi que le masque correspondant 
        pour image 1-2 et pour image 2-3.
        '''
        # Ici, on cherche à détecter les mêmes coordonnées de points 2D 
        # dans image_points_1 et image_points_2,
        # puis on applique un masque correspondant sur image_points_3.
        # Cela retourne 4 éléments afin de les utiliser ensuite dans la triangulation, PnP, etc.
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            # L’égalité stricte (==) peut poser des problèmes de précision, 
            # il serait préférable d’utiliser une tolérance de distance.

            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])
        # On applique un masque et on compresse image_points_2 et image_points_3 
        # en supprimant les indices trouvés.
        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(f"\n[Common Points] Between images => shape1={mask_array_1.shape}, shape2={mask_array_2.shape}")
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

class Sfm():
    def __init__(self, img_dir:str, downscale_factor:float = 2.0) -> None:
        '''
        Initialise un objet Sfm.
        '''
        self.img_obj = Image_loader(img_dir,downscale_factor)
        self.feature_extractor = FeatureExtractor()
        self.triangulator = Triangulator()
        self.pnp_solver = PnPSolver()
        self.reproj_calc = ReprojectionErrorCalculator()
        self.bundle_adjuster = BundleAdjuster(self.reproj_calc)
        self.ply_saver = PlySaver()
        self.common_finder = CommonPointsFinder()

    def __call__(self, enable_bundle_adjustment:boolean=True):#True ou False
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
        #  Matrice de projection
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4)) 
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        #image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        #image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))
        image_0 = cv2.imread(self.img_obj.image_list[0])
        image_1 = cv2.imread(self.img_obj.image_list[1])

        feature_0, feature_1 = self.feature_extractor.find_features(image_0, image_1)

        # Matrice essentielle : estimation par RANSAC.
 
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        # recoverPose renvoie R, t à partir de la matrice essentielle (il existe 4 solutions possibles).

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]       
        # Mise à jour de la pose en cumulant la rotation/translation sur la pose précédente.
 
        
        # R1​=RR0
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        #​ t1​=t0​+R0T​t
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())
        # P=K[R∣t]
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)
        # Triangulation pour obtenir un nuage de points 3D.
        feature_0, feature_1, points_3d = self.triangulator.triangulation(pose_0, pose_1, feature_0, feature_1)
        # Calcul de l'erreur de reprojection.
        error, points_3d = self.reproj_calc.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
        #ideally error < 1
        print("REPROJECTION ERROR: ", error)
        # PnP pour affiner la pose caméra et filtrer les outliers avec RANSAC.
 
        _, _, feature_1, points_3d, _ = self.pnp_solver.PnP(points_3d, feature_1, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2 
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.5
        errors = []

        for i in tqdm(range(total_images), desc="Processing frames", ascii=True, ncols=80):
            #image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            image_2 = cv2.imread(self.img_obj.image_list[i + 2])
            features_cur, features_2 = self.feature_extractor.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulator.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_finder.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.pnp_solver.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
            # Calcul de l'erreur de reprojection avant la triangulation suivante
            error, points_3d = self.reproj_calc.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity = 0)
        
            # Triangulation supplémentaire pour obtenir plus de points 3D à partir des vues successives
            cm_mask_0, cm_mask_1, points_3d = self.triangulator.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reproj_calc.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            # Si l'ajustement de faisceau est activé, on l'effectue ici. 
            # Attention : cela peut prendre beaucoup de temps.

            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjuster.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reproj_calc.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
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
            errors.append(error)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        plt.plot(range(len(errors)), errors, 'o-', color='blue')
        plt.xlabel("Frame Index")
        plt.ylabel("Error Value")
        plt.title("Reprojection Error Over Frames")
        plt.savefig("reprojection_error_plot.png", dpi=200)
        plt.show()
        
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.ply_saver.to_ply(self.img_obj.path, total_points, total_colors,self.img_obj.image_list)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')
    
if __name__ == '__main__':
    sfm = Sfm("Datasets\\fountain-P11")
    sfm()

