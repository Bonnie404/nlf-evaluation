# The code is adapted version of the code provided by the authors of the paper:

# @article{
# yin2025smplest,
# title={SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation},
# author={Yin, Wanqi and Cai, Zhongang and Wang, Ruisi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Mei, Haiyi and Wang, Yanjun and Pang, Hui En and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yamashita, Atsushi and Yang, Lei and Liu, Ziwei},
# journal={arXiv preprint arXiv:2501.09782},
# year={2025}
# }

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrender
import trimesh


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    kp_mask = np.copy(img)

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    kp_mask = np.copy(img)

    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_mesh(img, mesh_vertex, alpha=0.5, color_range=(0.2, 0.8)):
    if color_range[0] < 0 or color_range[1] > 1 or color_range[1] < color_range[0]:
        raise ValueError("color_range should be in [0, 1] and color_range[1] > color_range[0]")

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(color_range[0], color_range[1], len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    mask = np.copy(img)

    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def perspective_projection(vertices, cam_param):
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices


def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False, alpha=0.5):
    if mesh_as_vertices:
        vertices_2d = perspective_projection(vertices, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=alpha, radius=2, color=(0, 0, 255))
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.camera.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1], znear=0.05,
                                                  zfar=6000.0)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.3,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.4, 0.2, 0.1))

        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        cam_pose = pyrender2opencv @ np.eye(4)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=pyrender2opencv)
        scene.add(body_mesh)

        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                       viewport_height=img.shape[0],
                                       point_size=1.0)

        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        valid_mask = valid_mask * alpha
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

        img = (output_img * 255).astype(np.uint8)

    return img
