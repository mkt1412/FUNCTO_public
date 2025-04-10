import numpy as np
import cv2
import open3d as o3d
from shapely import point_on_surface

# depth_scale = 0.001 # for Kinect
# depth_scale = 0.00025  # for L515

def compute_point_cloud_no_add(rgb, depth,intrinsic, additional_points, depth_scale=0.001):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 641, 363

    points = []
    colors = []

    # (476, 480, 606) - (1280, 720, d)
    # import pdb;pdb.set_trace()
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            z = depth[v, u] * depth_scale  # for L515
            if z == 0 or z > 1:
            # if z == 0:
                continue  # Ignore invalid or distant depth values
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            color = rgb[v, u] / 255.0  # Normalize the color to [0, 1]
            points.append([x, y, z])
            colors.append(color)

    # loop through sphere center
    sphere_radius = 3
    sphere_color = [0, 1, 0]
    sphere_density = 1000

    # import pdb;pdb.set_trace()

    pts_3d = []
    for point in additional_points:
        sphere_center = point
        # TODO: compute the point in 3d
        x, y, z = sphere_center[0], sphere_center[1], sphere_center[2]
        # x = (x - cx) * z / fx / 1000.0
        # y = (y - cy) * z / fy / 1000.0
        # z = z / 1000.0
        x = ((x - cx) * z / fx) * depth_scale
        y = ((y - cy) * z / fy) * depth_scale
        z = z * depth_scale
        pts_3d.append([x, y, z])

        # if sphere_center[2] == 0:
        #     print("zero-depth point!")
        #     print(sphere_center)

        # skip zero center point
        if sphere_center[2] == 0:
            continue

        # for _ in range(sphere_density):
        #     phi = np.random.uniform(0, np.pi)
        #     theta = np.random.uniform(0, 2 * np.pi)
        #     r = sphere_radius * np.random.uniform(0, 1)**(1/3)

        #     x = r * np.sin(phi) * np.cos(theta) + sphere_center[0]
        #     y = r * np.sin(phi) * np.sin(theta) + sphere_center[1]
        #     z = r * np.cos(phi) + sphere_center[2]

        #     x = ((x - cx) * z / fx) * depth_scale
        #     y = ((y - cy) * z / fy) * depth_scale
        #     z = z * depth_scale

        #     points.append([x, y, z])
        #     colors.append(sphere_color)

    points = np.array(points)
    colors = np.array(colors)
    
    return points, colors, pts_3d

def compute_point_cloud(rgb, depth,intrinsic, additional_points, depth_scale=0.001):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 641, 363

    points = []
    colors = []

    # (476, 480, 606) - (1280, 720, d)
    # import pdb;pdb.set_trace()
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            z = depth[v, u] * depth_scale  # for L515
            if z == 0 or z > 1:
            # if z == 0:
                continue  # Ignore invalid or distant depth values
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            color = rgb[v, u] / 255.0  # Normalize the color to [0, 1]
            points.append([x, y, z])
            colors.append(color)

    # loop through sphere center
    sphere_radius = 3
    sphere_color = [0, 1, 0]
    sphere_density = 1000

    # import pdb;pdb.set_trace()

    pts_3d = []
    for point in additional_points:
        sphere_center = point
        # TODO: compute the point in 3d
        x, y, z = sphere_center[0], sphere_center[1], sphere_center[2]
        # x = (x - cx) * z / fx / 1000.0
        # y = (y - cy) * z / fy / 1000.0
        # z = z / 1000.0
        x = ((x - cx) * z / fx) * depth_scale
        y = ((y - cy) * z / fy) * depth_scale
        z = z * depth_scale
        pts_3d.append([x, y, z])

        # if sphere_center[2] == 0:
        #     print("zero-depth point!")
        #     print(sphere_center)

        # skip zero center point
        if sphere_center[2] == 0:
            continue

        for _ in range(sphere_density):
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2 * np.pi)
            r = sphere_radius * np.random.uniform(0, 1)**(1/3)

            x = r * np.sin(phi) * np.cos(theta) + sphere_center[0]
            y = r * np.sin(phi) * np.sin(theta) + sphere_center[1]
            z = r * np.cos(phi) + sphere_center[2]

            x = ((x - cx) * z / fx) * depth_scale
            y = ((y - cy) * z / fy) * depth_scale
            z = z * depth_scale

            points.append([x, y, z])
            colors.append(sphere_color)

    points = np.array(points)
    colors = np.array(colors)
    
    return points, colors, pts_3d


def get_depth_value(depth_img, x, y, inpaint=0):
    mask = (depth_img == 0).astype(np.uint8)
    depth_img = depth_img.astype(np.float32)
    if inpaint == 1:
        inpainted_depth = cv2.inpaint(depth_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    else:
        inpainted_depth = depth_img
    depth_value = inpainted_depth[int(x), int(y)]

    return depth_value, inpainted_depth

def compute_3d_points(intrinsic, additional_points, depth_scale=0.001):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 641, 363

    points = []
    colors = []

    # loop through sphere center
    sphere_radius = 3
    sphere_color = [0, 1, 0]
    sphere_density = 1000

    # import pdb;pdb.set_trace()

    pts_3d = []
    for point in additional_points:
        sphere_center = point
        # TODO: compute the point in 3d
        x, y, z = sphere_center[0], sphere_center[1], sphere_center[2]
        # x = (x - cx) * z / fx / 1000.0
        # y = (y - cy) * z / fy / 1000.0
        # z = z / 1000.0
        x = ((x - cx) * z / fx) * depth_scale
        y = ((y - cy) * z / fy) * depth_scale
        z = z * depth_scale
        pts_3d.append([x, y, z])

        if sphere_center[2] == 0:
            print("zero-depth point!")
            print(sphere_center)

        # for _ in range(sphere_density):
        #     phi = np.random.uniform(0, np.pi)
        #     theta = np.random.uniform(0, 2 * np.pi)
        #     r = sphere_radius * np.random.uniform(0, 1)**(1/3)

        #     x = r * np.sin(phi) * np.cos(theta) + sphere_center[0]
        #     y = r * np.sin(phi) * np.sin(theta) + sphere_center[1]
        #     z = r * np.cos(phi) + sphere_center[2]

        #     x = ((x - cx) * z / fx) * depth_scale
        #     y = ((y - cy) * z / fy) * depth_scale
        #     z = z * depth_scale

        #     points.append([x, y, z])
        #     colors.append(sphere_color)

    points = np.array(points)
    colors = np.array(colors)
    
    return points, colors, pts_3d

def vis_keypoints_point_cloud(rgb, depth, cam_matrix, vis_keypoints):
    points, colors, points_3d = compute_point_cloud(rgb, depth, cam_matrix, vis_keypoints)
    scene_point_cloud = o3d.geometry.PointCloud()
    scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Create small green spheres at each corner and the center of the bounding box
    spheres = []
    for point in points_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color
        spheres.append(sphere)

    # Combine the original point cloud and the spheres
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        # Convert the sphere to a point cloud for easier combination
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc

    combined_point_cloud = scene_point_cloud + spheres_combined

    return combined_point_cloud

def ransac_rigid_transform_3D(A, B, max_iterations=100000, tolerance=0.005, min_inliers_ratio=0.9):
    best_inliers = []
    best_T = None

    num_points = A.shape[1]
    scaled_tolerance = (num_points / 50) * tolerance

    for i in range(max_iterations):
        # Randomly sample a subset of correspondences (3 points minimum for a rigid transformation)
        # sample_indices = np.random.choice(num_points, size=3, replace=False)

        sample_num = int(num_points*0.4)
        # sample_num = int(num_points*0.5)
        sample_indices = np.random.choice(num_points, size=sample_num, replace=False)
        A_sample = A[:, sample_indices]
        B_sample = B[:, sample_indices]

        # Estimate transformation using the sample
        T = rigid_transform_3D(A_sample, B_sample)

        # Apply the transformation to all points in A
        A_transformed = (T[:3, :3] @ A) + T[:3, 3].reshape(3, 1)

        # Calculate the distance between the transformed A points and B points
        distances = np.linalg.norm(A_transformed - B, axis=0)

        # Determine inliers based on the tolerance
        # inliers = np.where(distances < tolerance)[0]
        inliers = np.where(distances < scaled_tolerance)[0]

        # Update the best transformation if the current one has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_T = T

        # Early exit if we have enough inliers
        if len(best_inliers) > (sample_num*min_inliers_ratio):
            print(f"iteration {i} RANSAC early exit")
            break

    # Recalculate the transformation using all inliers
    if len(best_inliers) > 0:
        best_T = rigid_transform_3D(A[:, best_inliers], B[:, best_inliers])
    
    return best_T, best_inliers

# from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# python version of this function in BundleSDF
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T

def align_set_B_to_A_v1(A, B):
    """
    version 1: (1) align function points and (2) manipulation planes
    """

    # Calculate normal vectors
    n_A = np.cross(A[1] - A[0], A[2] - A[0])
    n_B = np.cross(B[1] - B[0], B[2] - B[0])
    
    # Normalize the normal vectors
    n_A /= np.linalg.norm(n_A)
    n_B /= np.linalg.norm(n_B)
    
    # Calculate the axis of rotation (cross product) and angle
    r = np.cross(n_B, n_A)
    r /= np.linalg.norm(r)  # Normalize the rotation axis
    theta = np.arccos(np.clip(np.dot(n_B, n_A), -1.0, 1.0))  # Angle between the normals
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # Align the first point
    translation = A[0] - B[0]
    B_prime = B + translation
    
    # Apply the rotation to the translated points
    B_aligned = np.dot(B_prime - A[0], R) + A[0]
    
    return B_aligned, R

def align_set_B_to_A_v2(A, B):
    """
    version 2: (1) align A[0] and B[0], (2) align planes, and (3) align axes
    """

    # import pdb;pdb.set_trace()

    # Step 1: Align A[0] with B[0]
    translation = A[0] - B[0]
    B_translated = B + translation
    
    # Step 2: Align the plane normals
    n_A = np.cross(A[1] - A[0], A[2] - A[0])
    n_B = np.cross(B_translated[1] - B_translated[0], B_translated[2] - B_translated[0])
    
    # Normalize the normal vectors
    n_A /= np.linalg.norm(n_A)
    n_B /= np.linalg.norm(n_B)
    
    # Calculate the axis of rotation and angle between the normals
    r = np.cross(n_B, n_A)
    r /= np.linalg.norm(r)  # Normalize the rotation axis
    theta = np.arccos(np.clip(np.dot(n_B, n_A), -1.0, 1.0))  # Angle between the normals

    # TODO: fix this later
    theta = -theta
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R_plane = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # Apply the rotation to align the plane
    B_aligned_plane = np.dot(B_translated - B_translated[0], R_plane) + B_translated[0]
    # B_aligned_plane = np.dot(R_plane, (B_translated - B_translated[0]).T).T + B_translated[0]

    # Step 3: Align the edges A[0]-A[1] with B[0]-B[1]
    edge_A = A[1] - A[0]
    edge_B = B_aligned_plane[1] - B_aligned_plane[0]
    
    # Normalize the edges
    edge_A /= np.linalg.norm(edge_A)
    edge_B /= np.linalg.norm(edge_B)
    
    # Calculate the axis of rotation and angle to align the edges
    axis_edge = np.cross(edge_B, edge_A)
    axis_edge /= np.linalg.norm(axis_edge)
    theta_edge = np.arccos(np.clip(np.dot(edge_B, edge_A), -1.0, 1.0))

    # TODO: fix this later
    theta_edge = -theta_edge
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K_edge = np.array([[0, -axis_edge[2], axis_edge[1]],
                       [axis_edge[2], 0, -axis_edge[0]],
                       [-axis_edge[1], axis_edge[0], 0]])
    R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    
    # Apply the rotation to align the edges
    B_final = np.dot(B_aligned_plane - B_aligned_plane[0], R_edge) + B_aligned_plane[0]

    
    return B_final

def align_set_B_to_A_v3_grasp(A, B):
    """
    version 2: (1) align A[0] and B[0], (2) align planes, and (3) align axes
    """

    # import pdb;pdb.set_trace()
    A_func, A_center, A_grasp = A
    B_func, B_center, B_grasp = B

    A = [A_grasp, A_func, A_center]
    B = [B_grasp, B_func, B_center]

    # Step 1: Align A[0] with B[0]
    translation = A[0] - B[0]
    B_translated = B + translation
    
    # Step 2: Align the plane normals
    n_A = np.cross(A[1] - A[0], A[2] - A[0])
    n_B = np.cross(B_translated[1] - B_translated[0], B_translated[2] - B_translated[0])
    
    # Normalize the normal vectors
    n_A /= np.linalg.norm(n_A)
    n_B /= np.linalg.norm(n_B)
    
    # Calculate the axis of rotation and angle between the normals
    r = np.cross(n_B, n_A)
    r /= np.linalg.norm(r)  # Normalize the rotation axis
    theta = np.arccos(np.clip(np.dot(n_B, n_A), -1.0, 1.0))  # Angle between the normals

    # TODO: fix this later
    theta = -theta
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R_plane = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # Apply the rotation to align the plane
    B_aligned_plane = np.dot(B_translated - B_translated[0], R_plane) + B_translated[0]
    # B_aligned_plane = np.dot(R_plane, (B_translated - B_translated[0]).T).T + B_translated[0]

    # Step 3: Align the edges A[0]-A[1] with B[0]-B[1]
    edge_A = A[1] - A[0]
    edge_B = B_aligned_plane[1] - B_aligned_plane[0]
    
    # Normalize the edges
    edge_A /= np.linalg.norm(edge_A)
    edge_B /= np.linalg.norm(edge_B)
    
    # Calculate the axis of rotation and angle to align the edges
    axis_edge = np.cross(edge_B, edge_A)
    axis_edge /= np.linalg.norm(axis_edge)
    theta_edge = np.arccos(np.clip(np.dot(edge_B, edge_A), -1.0, 1.0))

    # TODO: fix this later
    theta_edge = -theta_edge

    B_final_list = []
    T_final_list = []
    # for theta_offset in [0]
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K_edge = np.array([[0, -axis_edge[2], axis_edge[1]],
                    [axis_edge[2], 0, -axis_edge[0]],
                    [-axis_edge[1], axis_edge[0], 0]])
    # R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    
    # Apply the rotation to align the edges
    B_final = np.dot(B_aligned_plane - B_aligned_plane[0], R_edge) + B_aligned_plane[0]

    # Step 4: additioinal rotation for refinement
    n_B_final = np.cross(A[1] - A[0], A[2] - A[0])

    n_B_final /= np.linalg.norm(n_B_final)  # Normalize the normal vector

    for theta_normal in [-45, -30, -10, 0, 10, 30, 45]:
    # for theta_normal in [-35, -15, 0, 15, 35]:
    # for theta_normal in [0]:

        theta_normal = np.radians(theta_normal)

        K_normal = np.array([[0, -n_B_final[2], n_B_final[1]],
                            [n_B_final[2], 0, -n_B_final[0]],
                            [-n_B_final[1], n_B_final[0], 0]])

        R_normal = np.eye(3) + np.sin(theta_normal) * K_normal + (1 - np.cos(theta_normal)) * np.dot(K_normal, K_normal)

        B_final_rotated = np.dot(B_final - B_final[0], R_normal) + B_final[0]

        Br_grasp, Br_func, Br_center = B_final_rotated
        B_final_rotated = [Br_func, Br_center, Br_grasp]

        B_final_list.append(B_final_rotated)
        T_final_list.append([translation, R_plane, R_edge, R_normal])

    return B_final_list, T_final_list

def align_set_B_to_A_v3_center(A, B):
    """
    version 2: (1) align A[0] and B[0], (2) align planes, and (3) align axes
    """

    # import pdb;pdb.set_trace()
    A_func, A_center, A_grasp = A
    B_func, B_center, B_grasp = B

    A = [A_center, A_func, A_grasp]
    B = [B_center, B_func, B_grasp]

    # Step 1: Align A[0] with B[0]
    translation = A[0] - B[0]
    B_translated = B + translation
    
    # Step 2: Align the plane normals
    n_A = np.cross(A[1] - A[0], A[2] - A[0])
    n_B = np.cross(B_translated[1] - B_translated[0], B_translated[2] - B_translated[0])
    
    # Normalize the normal vectors
    n_A /= np.linalg.norm(n_A)
    n_B /= np.linalg.norm(n_B)
    
    # Calculate the axis of rotation and angle between the normals
    r = np.cross(n_B, n_A)
    r /= np.linalg.norm(r)  # Normalize the rotation axis
    theta = np.arccos(np.clip(np.dot(n_B, n_A), -1.0, 1.0))  # Angle between the normals

    # TODO: fix this later
    theta = -theta
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R_plane = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # Apply the rotation to align the plane
    B_aligned_plane = np.dot(B_translated - B_translated[0], R_plane) + B_translated[0]
    # B_aligned_plane = np.dot(R_plane, (B_translated - B_translated[0]).T).T + B_translated[0]

    # Step 3: Align the edges A[0]-A[1] with B[0]-B[1]
    edge_A = A[1] - A[0]
    edge_B = B_aligned_plane[1] - B_aligned_plane[0]
    
    # Normalize the edges
    edge_A /= np.linalg.norm(edge_A)
    edge_B /= np.linalg.norm(edge_B)
    
    # Calculate the axis of rotation and angle to align the edges
    axis_edge = np.cross(edge_B, edge_A)
    axis_edge /= np.linalg.norm(axis_edge)
    theta_edge = np.arccos(np.clip(np.dot(edge_B, edge_A), -1.0, 1.0))

    # TODO: fix this later
    theta_edge = -theta_edge

    B_final_list = []
    T_final_list = []
    # for theta_offset in [0]
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K_edge = np.array([[0, -axis_edge[2], axis_edge[1]],
                    [axis_edge[2], 0, -axis_edge[0]],
                    [-axis_edge[1], axis_edge[0], 0]])
    # R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    
    # Apply the rotation to align the edges
    B_final = np.dot(B_aligned_plane - B_aligned_plane[0], R_edge) + B_aligned_plane[0]

    # Step 4: additioinal rotation for refinement
    n_B_final = np.cross(A[1] - A[0], A[2] - A[0])

    n_B_final /= np.linalg.norm(n_B_final)  # Normalize the normal vector

    for theta_normal in [-45, -30, -10, 0, 10, 30, 45]:
    # for theta_normal in [45, 0, -10, 30, -30, 10, -45]:
    # for theta_normal in [35, -15, 15, 0, -35]:
    # for theta_normal in [0]:

        theta_normal = np.radians(theta_normal)

        K_normal = np.array([[0, -n_B_final[2], n_B_final[1]],
                            [n_B_final[2], 0, -n_B_final[0]],
                            [-n_B_final[1], n_B_final[0], 0]])

        R_normal = np.eye(3) + np.sin(theta_normal) * K_normal + (1 - np.cos(theta_normal)) * np.dot(K_normal, K_normal)

        B_final_rotated = np.dot(B_final - B_final[0], R_normal) + B_final[0]

        Br_center, Br_func, Br_grasp = B_final_rotated
        B_final_rotated = [Br_func, Br_center, Br_grasp]

        B_final_list.append(B_final_rotated)
        T_final_list.append([translation, R_plane, R_edge, R_normal])

    return B_final_list, T_final_list


def align_set_B_to_A_v3(A, B):
    """
    version 2: (1) align A[0] and B[0], (2) align planes, and (3) align axes

    A: function, B: center, C: grasp
    """

    # import pdb;pdb.set_trace()

    # Step 1: Align A[0] with B[0]
    translation = A[0] - B[0]
    B_translated = B + translation
    
    # Step 2: Align the plane normals
    n_A = np.cross(A[1] - A[0], A[2] - A[0])
    n_B = np.cross(B_translated[1] - B_translated[0], B_translated[2] - B_translated[0])
    
    # Normalize the normal vectors
    n_A /= np.linalg.norm(n_A)
    n_B /= np.linalg.norm(n_B)
    
    # Calculate the axis of rotation and angle between the normals
    r = np.cross(n_B, n_A)
    r /= np.linalg.norm(r)  # Normalize the rotation axis
    theta = np.arccos(np.clip(np.dot(n_B, n_A), -1.0, 1.0))  # Angle between the normals

    # TODO: fix this later
    theta = -theta
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R_plane = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # Apply the rotation to align the plane
    B_aligned_plane = np.dot(B_translated - B_translated[0], R_plane) + B_translated[0]
    # B_aligned_plane = np.dot(R_plane, (B_translated - B_translated[0]).T).T + B_translated[0]

    # Step 3: Align the edges A[0]-A[1] with B[0]-B[1]
    edge_A = A[1] - A[0]
    edge_B = B_aligned_plane[1] - B_aligned_plane[0]
    
    # Normalize the edges
    edge_A /= np.linalg.norm(edge_A)
    edge_B /= np.linalg.norm(edge_B)
    
    # Calculate the axis of rotation and angle to align the edges
    axis_edge = np.cross(edge_B, edge_A)
    axis_edge /= np.linalg.norm(axis_edge)
    theta_edge = np.arccos(np.clip(np.dot(edge_B, edge_A), -1.0, 1.0))

    # TODO: fix this later
    theta_edge = -theta_edge

    B_final_list = []
    T_final_list = []
    # for theta_offset in [0]
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    K_edge = np.array([[0, -axis_edge[2], axis_edge[1]],
                    [axis_edge[2], 0, -axis_edge[0]],
                    [-axis_edge[1], axis_edge[0], 0]])
    # R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    R_edge = np.eye(3) + np.sin(theta_edge) * K_edge + (1 - np.cos(theta_edge)) * np.dot(K_edge, K_edge)
    
    # Apply the rotation to align the edges
    B_final = np.dot(B_aligned_plane - B_aligned_plane[0], R_edge) + B_aligned_plane[0]

    # Step 4: additioinal rotation for refinement
    n_B_final = np.cross(A[1] - A[0], A[2] - A[0])

    n_B_final /= np.linalg.norm(n_B_final)  # Normalize the normal vector

    for theta_normal in [-40, -30, -10, 0, 10, 30, 40]:
    # for theta_normal in [-30, -10, 0, 10, 30]:
    # for theta_normal in [0]:

        theta_normal = np.radians(theta_normal)

        K_normal = np.array([[0, -n_B_final[2], n_B_final[1]],
                            [n_B_final[2], 0, -n_B_final[0]],
                            [-n_B_final[1], n_B_final[0], 0]])

        R_normal = np.eye(3) + np.sin(theta_normal) * K_normal + (1 - np.cos(theta_normal)) * np.dot(K_normal, K_normal)

        B_final_rotated = np.dot(B_final - B_final[0], R_normal) + B_final[0]

        B_final_list.append(B_final_rotated)
        T_final_list.append([translation, R_plane, R_edge, R_normal])

    return B_final_list, T_final_list

def point_transform(pt_3d, trans):
    pt_3d_hom = np.append(np.array(pt_3d), 1)
    transformed_pt_3d_hom = trans @ pt_3d_hom
    transformed_pt_3d = transformed_pt_3d_hom[:3]

    return transformed_pt_3d

def calculate_rotation_matrix_z(point1, point2):
    """
    Calculate the Z-axis rotation matrix to align point2 with point1.
    """
    # Project the points onto the XY-plane (ignore the Z component)
    p1 = point1[:2]  # (x1, y1)
    p2 = point2[:2]  # (x2, y2)
    
    # Calculate the angles of each point from the origin (0, 0)
    angle1 = np.arctan2(p1[1], p1[0])  # Angle of point1
    angle2 = np.arctan2(p2[1], p2[0])  # Angle of point2

    # Calculate the rotation angle needed (difference between angles)
    rotation_angle = angle1 - angle2

    # Create the Z-axis rotation matrix using the calculated angle
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    return rotation_matrix


