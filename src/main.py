import argparse
import bisect
import os
import os.path as osp
import time
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import json
import yaml
from smplx import SMPL
from scipy.spatial.transform import Rotation as R
from src.visualization_utils import render_mesh
import csv
import concurrent.futures


def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and returns it as a dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="3D Pose Inference Script")
    parser.add_argument("--config", type=str, default=osp.join(os.path.dirname(os.getcwd()), "configs", "config.yaml"),
                        help="Path to YAML configuration file")
    return parser.parse_args()


def get_col_name_to_csv_indices():
    """
    Returns a dictionary that contains, for each joint, the indices of the columns (x, y, z, p) in the CSV file.
    You must ensure that the indices here exactly match the columns in your CSV.
    """
    return {
        "nose": (2, 3, 4, 5),
        "lElbow": (6, 7, 8, 9),
        "lWrist": (10, 11, 12, 13),
        "rHeel": (14, 15, 16, 17),
        "rHip": (18, 19, 20, 21),
        "rSmallToe": (22, 23, 24, 25),
        "neck": (26, 27, 28, 29),
        "lSmallToe": (30, 31, 32, 33),
        "rWrist": (34, 35, 36, 37),
        "rAnkle": (38, 39, 40, 41),
        "lHip": (42, 43, 44, 45),
        "lHeel": (46, 47, 48, 49),
        "lKnee": (50, 51, 52, 53),
        "lEye": (54, 55, 56, 57),
        "midHip": (58, 59, 60, 61),
        "background": (62, 63, 64, 65),  # Kann ignoriert werden, falls nicht benötigt
        "lEar": (66, 67, 68, 69),
        "rElbow": (70, 71, 72, 73),
        "rShoulder": (74, 75, 76, 77),
        "rKnee": (78, 79, 80, 81),
        "lShoulder": (82, 83, 84, 85),
        "lBigToe": (86, 87, 88, 89),
        "rEye": (90, 91, 92, 93),
        "rEar": (94, 95, 96, 97),
        "rBigToe": (98, 99, 100, 101),
        "lAnkle": (102, 103, 104, 105),
    }


openpose_order = [
    "nose",  # 0
    "neck",  # 1
    "rShoulder",  # 2
    "rElbow",  # 3
    "rWrist",  # 4
    "lShoulder",  # 5
    "lElbow",  # 6
    "lWrist",  # 7
    "midHip",  # 8
    "rHip",  # 9
    "rKnee",  # 10
    "rAnkle",  # 11
    "lHip",  # 12
    "lKnee",  # 13
    "lAnkle",  # 14
    "rEye",  # 15
    "lEye",  # 16
    "rEar",  # 17
    "lEar",  # 18
    "lBigToe",  # 19
    "lSmallToe",  # 20
    "lHeel",  # 21
    "rBigToe",  # 22
    "rSmallToe",  # 23
    "rHeel",  # 24
]


def read_pose_csv(filename):
    """
    Reads the CSV file and returns a dictionary where for each frame_id the values for 'timestamp' and a list of joints (joints_list) are stored.

    """
    col_name_to_indices = get_col_name_to_csv_indices()

    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            frame_id = int(row[0])
            timestamp = float(row[1])

            joints_list = []
            for joint_name in openpose_order:
                x_idx, y_idx, z_idx, p_idx = col_name_to_indices[joint_name]

                x_val = float(row[x_idx])
                y_val = float(row[y_idx])
                z_val = float(row[z_idx])
                p_val = float(row[p_idx])

                joints_list.append([x_val, y_val, z_val, p_val])

            data[frame_id] = {
                "timestamp": timestamp,
                "joints": joints_list
            }

    return data


def smpl_to_openpose(smpl_joints: np.ndarray):
    """
    Converts the directly available SMPL joints (24 points, index 0..23)
    to the OpenPose BODY_25 format (25 joints), WITHOUT estimating missing joints.
    Instead, joints that cannot be mapped are set to (0,0,0).

    Additionally returns a mask (25,) that has value 1 at indices
    of defined joints and 0 otherwise.

    Parameters:
    ----------
    smpl_joints : np.ndarray, Shape (24,3)
      SMPL joint coordinates:
        Index -> Name
         0:  Pelvis        1:  L_Hip        2:  R_Hip       3:  Spine1
         4:  L_Knee        5:  R_Knee       6:  Spine2      7:  L_Ankle
         8:  R_Ankle       9:  Spine3       10: L_Foot      11: R_Foot
         12: Neck          13: L_Collar     14: R_Collar    15: Head
         16: L_Shoulder    17: R_Shoulder   18: L_Elbow     19: R_Elbow
         20: L_Wrist       21: R_Wrist      22: L_Hand      23: R_Hand

    Returns:
    --------
    openpose_joints : np.ndarray, Shape (25,3)
      3D coordinates in BODY_25 format, where joints that cannot be mapped
      are set to (0,0,0).

    mask : np.ndarray, Shape (25,)
      Binary mask, 1 = joint is defined/mapped, 0 = joint not defined.
"""

    # OpenPose BODY_25-Indices:
    #  0: Nose,       1: Neck,         2: RShoulder,  3: RElbow,      4: RWrist,
    #  5: LShoulder,  6: LElbow,       7: LWrist,     8: MidHip,      9: RHip,
    # 10: RKnee,     11: RAnkle,      12: LHip,      13: LKnee,      14: LAnkle,
    # 15: REye,      16: LEye,        17: REar,      18: LEar,       19: LBigToe,
    # 20: LSmallToe, 21: LHeel,       22: RBigToe,   23: RSmallToe,  24: RHeel

    direct_mapping = {
        1: 12,  # OP-Neck       -> SMPL[12] (Neck)
        2: 17,  # OP-RShoulder  -> SMPL[17] (R_Shoulder)
        3: 19,  # OP-RElbow     -> SMPL[19] (R_Elbow)
        4: 21,  # OP-RWrist     -> SMPL[21] (R_Wrist)
        5: 16,  # OP-LShoulder  -> SMPL[16] (L_Shoulder)
        6: 18,  # OP-LElbow     -> SMPL[18] (L_Elbow)
        7: 20,  # OP-LWrist     -> SMPL[20] (L_Wrist)
        8: 0,  # OP-MidHip     -> SMPL[0]  (Pelvis)
        9: 2,  # OP-RHip       -> SMPL[2]  (R_Hip)
        10: 5,  # OP-RKnee      -> SMPL[5]  (R_Knee)
        11: 8,  # OP-RAnkle     -> SMPL[8]  (R_Ankle)
        12: 1,  # OP-LHip       -> SMPL[1]  (L_Hip)
        13: 4,  # OP-LKnee      -> SMPL[4]  (L_Knee)
        14: 7,  # OP-LAnkle     -> SMPL[7]  (L_Ankle)
        19: 10,  # OP-LBigToe    -> SMPL[10] (L_Foot)
        22: 11,  # OP-RBigToe    -> SMPL[11] (R_Foot)
    }

    openpose_joints = np.zeros((25, 3), dtype=smpl_joints.dtype)
    mask = np.zeros((25,), dtype=np.int32)

    for op_idx, smpl_idx in direct_mapping.items():
        openpose_joints[op_idx] = smpl_joints[smpl_idx]
        mask[op_idx] = 1  # definiert

    return openpose_joints, mask


def transform_points_to_camera(points_world: np.ndarray, camera_extrinsic: np.ndarray):
    """
    Transforms the 3D points (N,3) from world coordinates into camera coordinates:
    X_cam = camera_extrinsic * X_world.
    Returns an array (N,3) in camera coordinates.
    """
    N = points_world.shape[0]
    ones = np.ones((N, 1), dtype=points_world.dtype)
    points_world_h = np.hstack([points_world, ones])  # (N,4)

    points_cam_h = (camera_extrinsic @ points_world_h.T).T  # => (N,4)
    # x_cam, y_cam, z_cam (Spalte 3 = 1)
    points_cam = points_cam_h[:, :3] / points_cam_h[:, 3:4]
    return points_cam  # (N,3)


def project_points_cam_to_2d(points_cam: np.ndarray,
                             camera_intrinsic: np.ndarray,
                             distortion_coeffs: np.ndarray = None):
    """
    Projects 3D points (N,3) from the camera coordinate system (i.e. WITHOUT extrinsic parameters)
    into the image using the intrinsic parameters and distortion.
    Returns (N,2) pixel coordinates.
    """
    N = points_cam.shape[0]
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]

    points_cam_for_opencv = np.stack([x_cam / z_cam,
                                      y_cam / z_cam,
                                      np.zeros_like(z_cam)], axis=-1)  # (N,3)
    points_cam_for_opencv = points_cam_for_opencv.reshape(N, 1, 3)

    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    camera_matrix_cv2 = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    if distortion_coeffs is None:
        dist = np.zeros((5,), dtype=np.float64)
    else:
        dist = distortion_coeffs.astype(np.float64)

    rvec = np.zeros((3,))
    tvec = np.zeros((3,))
    points_2d, _ = cv2.projectPoints(points_cam_for_opencv, rvec, tvec,
                                     camera_matrix_cv2, dist)
    points_2d = np.squeeze(points_2d, axis=1)
    return points_2d


def read_timestamps_file(timestamps_file):
    with open(timestamps_file, 'r') as f:
        lines = f.read().splitlines()
    video_timestamps = [float(line.strip()) for line in lines if line.strip()]
    return video_timestamps


def find_closest_frame_index(video_timestamps, annotation_timestamp):
    pos = bisect.bisect_left(video_timestamps, annotation_timestamp)
    if pos == 0:
        return 0
    if pos >= len(video_timestamps):
        return len(video_timestamps) - 1
    before = pos - 1
    after = pos
    diff_before = abs(video_timestamps[before] - annotation_timestamp)
    diff_after = abs(video_timestamps[after] - annotation_timestamp)
    return before if diff_before <= diff_after else after


def read_frame_for_timestamp(cap, ts_sec):
    """
    Seeks the video to the approximate timestamp and reads one frame.
    Returns image or None upon failure.
    """
    # Convert seconds to milliseconds for OpenCV
    ms_position = ts_sec * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, ms_position)

    # Attempt to read one frame
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    return frame


def video_to_images(video_path, images_path, timestamps, workers_num=16):
    """
    Converts a video to images according to the provided timestamps and saves them in the specified folder.
    The frames are retrieved based on the timestamps array. For each timestamp, the code seeks the video
    to the corresponding position (in milliseconds), grabs the frame, and writes the resulting image to disk.

    Args:
        video_path (str): Path to the video file.
        images_path (str): Path to the folder where images will be saved.
        timestamps (List[float]): List of timestamps (in seconds or milliseconds, depending on usage)
            for the frames to be extracted.

    Returns:
        int: Number of frames successfully extracted and saved as images.
    """

    if not os.path.exists(images_path):
        os.makedirs(images_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0

    def write_image(idx, image):
        """Write a single image to disk, named as %06d.jpg."""
        out_path = os.path.join(images_path, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, image)

    num_written = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_num) as executor:
        for idx, ts in enumerate(timestamps):
            image = read_frame_for_timestamp(cap, ts)
            if image is not None:
                future_to_frame = {
                    executor.submit(write_image, idx, image): (idx, image)
                }
            else:
                print(f"Warning: Could not read frame for timestamp {ts:.3f} sec.")
                continue

            if idx % 1000 == 0:
                for fut in concurrent.futures.as_completed(future_to_frame):
                    fut.result()
                    num_written += 1

    cap.release()
    print(f"Extracted {num_written} frames and wrote them to '{images_path}'.")
    return num_written


def get_camera_calibration(file_path):
    with open(file_path, 'r') as f:
        calibration_data = json.load(f)

    camera_extrinsics = calibration_data['extrinsics']
    camera_translation = camera_extrinsics['translation']
    rot_vec = camera_extrinsics['rotation']
    quat = [rot_vec['x'], rot_vec['y'], rot_vec['z'], rot_vec['w']]
    rotation_matrix = R.from_quat(quat).as_matrix()
    camera_extrinsic_matrix = np.array([
        [rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], camera_translation['x']],
        [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], camera_translation['y']],
        [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], camera_translation['z']],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    focal_length = calibration_data['intrinsics']['focallength']
    principal_point = calibration_data['intrinsics']['principal_point']

    camera_intrinsic_matrix = np.array([
        [focal_length['fx'], 0, principal_point['cx']],
        [0, focal_length['fy'], principal_point['cy']],
        [0, 0, 1]
    ], dtype=np.float32)

    distortion = calibration_data['intrinsics']['distortion']
    distortion_coeffs = np.array([
        distortion["k1"],
        distortion["k2"],
        distortion["p1"],
        distortion["p2"],
        distortion["k3"]
    ], dtype=np.float32)

    return camera_extrinsic_matrix, camera_intrinsic_matrix, distortion_coeffs


def frames_to_video(input_path, output_path, video_timestamps):
    """
    Creates a video from the images in input_path. The images are named, for example, '150.jpg' ... '200.jpg',
    where 150 and 200 are indices in the 'video_timestamps' list.

    Procedure:
      1. Determine all available JPG files in the folder.
      2. Sort them by their index number (filename without '.jpg').
      3. Determine min_index and max_index.
      4. Calculate total_time = video_timestamps[max_index] - video_timestamps[min_index].
      5. Calculate fps = (number_of_images - 1) / total_time (if total_time > 0; if total_time is 0, use a fallback FPS).
      6. Create the video in 'output_path'.
    """

    filenames = [f for f in os.listdir(input_path) if f.endswith('.jpg')]
    if not filenames:
        print("No images in directory!")
        return

    def parse_index(fname):
        base, _ = osp.splitext(fname)  # '150'
        return int(base)

    filenames.sort(key=parse_index)

    indices = [parse_index(fname) for fname in filenames]

    min_idx = min(indices)
    max_idx = max(indices)

    if len(indices) == 1:
        print("Only one image was found!")
        return

    # Gesamte Zeitspanne in den Timestamps
    start_time = video_timestamps[min_idx]
    end_time = video_timestamps[max_idx]
    total_time = end_time - start_time

    # FPS berechnen
    n_frames = len(indices)
    if total_time > 0:
        fps = (n_frames - 1) / total_time
    else:
        raise ValueError("Total time is zero, cannot calculate FPS.")

    print(f"Create a video from {min_idx} to {max_idx}, total_time={total_time:.3f} s, fps={fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    for fname in filenames:
        img_path = osp.join(input_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}.")
            continue

        if out is None:
            h, w, c = img.shape
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(img)

    if out is not None:
        out.release()
        print(f"Video saved as {output_path}")
    else:
        print("No images available - video not created")


def save_mpjpe_to_csv(mpjpe_values, output_file):
    """
    Saves MPJPE values (Mean Per Joint Position Error) to a CSV file.

    Parameters:
      - mpjpe_values: A dictionary with frame IDs as keys and MPJPE values in mm as values.
      - output_file: Path to the output file where the MPJPE values will be written.
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['frame_id', 'mpjpe_mm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for frame_id, mpjpe in mpjpe_values.items():
                writer.writerow({'frame_id': frame_id, 'mpjpe_mm': mpjpe})
        print(f"MPJPE-Werte erfolgreich in {output_file} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der MPJPE-Werte: {e}")


def process_single_frame(idx, annotations, video_timestamps, image_folder):
    """
    Performs the same operation as in the original loop:
      1. Check if 'idx' exists in 'annotations'.
      2. Convert the timestamp to the closest frame index.
      3. Load the image.
      4. Return the image path and frame index.

    Returns None if the image could not be loaded.
    """
    if idx not in annotations:
        return None

    annotation_timestamp = annotations[idx]["timestamp"]
    frame_idx = find_closest_frame_index(video_timestamps, annotation_timestamp)
    image_path = osp.join(image_folder, f"{frame_idx:06d}.jpg")

    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img_bgr = cv2.imread(image_path)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {e}")

    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return idx, image_path, frame_idx


def parallel_image_loading(image_start, image_end, annotations, video_timestamps, image_folder, max_workers=8):
    """
    Loads the images in parallel for the range [image_start, image_end).
    returns:
      - image_paths
      - frame_indices
    Optionally, it can also return the loaded images in a list if needed.
    """
    image_paths = [0] * (image_end - image_start)
    frame_indices = [0] * (image_end - image_start)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_frame, idx, annotations, video_timestamps, image_folder
            ): idx
            for idx in range(image_start, image_end)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            result = future.result()
            if result is not None:
                idx, image_path, frame_idx = result
                image_paths[idx - image_start] = image_path
                frame_indices[idx - image_start] = frame_idx

    return image_paths, frame_indices


def to_full_path(cwd, relative_path):
    """
    Converts a relative path to an absolute path based on the current working directory.
    """
    if not relative_path.startswith(cwd):
        return osp.join(cwd, relative_path)
    return relative_path

def main():
    args = parse_args()
    config = load_config(args.config)

    cwd = osp.dirname(osp.dirname(os.path.abspath(__file__)))
    annotations_file_csv = to_full_path(cwd, config['files']['annotations_file_csv'])
    video_file = to_full_path(cwd, config['files']['video_file'])
    timestamps_file = to_full_path(cwd, config['files']['timestamps_file'])
    calibration_file = to_full_path(cwd, config['files']['calibration_file'])
    output_frames_folder = to_full_path(cwd, config['files']['output_frames_folder'])
    output_video_folder = to_full_path(cwd, config['files']['output_video_folder'])
    image_folder = to_full_path(cwd, config['files']['image_folder'])

    nlf_model_file_path = to_full_path(cwd, config['nlf']['nlf_model_path'])

    smpl_model_path = to_full_path(cwd, config['smpl']['smpl_model_path'])

    image_start = config['video_processing']['image_start']
    length = config['video_processing']['length']
    batch_size = config['video_processing']['batch_size']
    image_end = image_start + length

    max_workers = config['parallel']['max_workers']

    annotations = read_pose_csv(annotations_file_csv)
    video_timestamps = read_timestamps_file(timestamps_file)
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Video could not be opened.")
    else:
        print("Number of frames in the video:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    camera_extrinsic_matrix, camera_intrinsic_matrix, distortion_coeffs = get_camera_calibration(calibration_file)

    if config['video_processing']['extract_frames']:
        video_to_images(video_file, image_folder,
                        [val['timestamp'] - video_timestamps[0] for _, val in annotations.items()],
                        workers_num=max_workers)

    # Load SMPL model
    smpl_model = SMPL(smpl_model_path, gender='MALE')

    frames_batch = []
    annotations_list = list(annotations.items())
    for frame_idx in range(image_start, image_end, batch_size):
        frames_batch.append(annotations_list[frame_idx:min(frame_idx + batch_size, len(annotations))])

    mpjpe_values = {}

    for annotations_batch in frames_batch:
        bgr_images_batch = [read_frame_for_timestamp(cap, val['timestamp'] - video_timestamps[0]) for _, val in
                            annotations_batch]
        batch = torch.stack([(F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) * 255).to(torch.uint8) for img in bgr_images_batch], dim=0).cuda()
        print("Starting the inference, Batch-Size =", batch.shape[0])
        start_time = time.time()
        try:
            with torch.inference_mode():
                nlf_model = torch.jit.load(nlf_model_file_path).cuda().eval()
                pred = nlf_model.detect_smpl_batched(batch,
                                                     intrinsic_matrix=torch.tensor(camera_intrinsic_matrix).unsqueeze(
                                                         0).cuda(),
                                                     distortion_coeffs=torch.tensor(distortion_coeffs).unsqueeze(
                                                         0).cuda(),
                                                     )
                torch.cuda.synchronize()

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.3f} s")
            B = batch.shape[0]
            pred['global_orient'] = [p[:, :3] for p in pred['pose']]
            pred['pose'] = [p[:, 3:] for p in pred['pose']]
            joints3d_b = pred['joints3d']

            if len(joints3d_b) == 0:
                print("No persons found in the batch.")
                return

            for b in range(B):  # for each image in this batch
                vis_image = bgr_images_batch[b]
                text_pos = (int(vis_image.shape[1] - 220), 50)
                frame_idx = annotations_batch[b][0]
                if pred['joints3d'][b].shape[0] == 0:
                    print("No persons found in the image.")
                    cv2.putText(vis_image,
                                f"MPJPE: N/A",
                                text_pos,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.0,
                                color=(0, 0, 255),
                                thickness=2)
                    output_path = osp.join(output_frames_folder, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(output_path, vis_image)
                    continue

                joints3d_b = pred['joints3d'][b]
                joints2d_b = pred['joints2d'][b]
                boxes_b = pred['boxes'][b]
                vertices3d_b = pred['vertices3d'][b]

                # TODO: vectorize the for-loop to render all persons at once
                for pid in range(len(boxes_b)):
                    # sanity check
                    if np.allclose(joints2d_b[pid].cpu().numpy(),
                                   project_points_cam_to_2d(joints3d_b[pid].cpu().numpy(),
                                                            camera_intrinsic_matrix,
                                                            distortion_coeffs)):
                        print("2D joints and projected 3D joints are the same")

                    # the pyrender camera does not support distortion coefficients, therefore we need to
                    # e.g. adjust the 3D points beforehand,
                    # so they are also taken into account (otherwise the mistake is quite large)
                    u_dist = project_points_cam_to_2d(vertices3d_b[pid].cpu().numpy(),
                                                      camera_intrinsic_matrix,
                                                      distortion_coeffs)
                    vertices_3d_cam_baked = vertices3d_b[pid].cpu().numpy()
                    z_vals = vertices_3d_cam_baked[:, 2]
                    shifted_2d = u_dist - np.array([camera_intrinsic_matrix[0, 2], camera_intrinsic_matrix[1, 2]])
                    vertices_3d_cam_baked[:, 0:2] = shifted_2d * (
                            z_vals[:, None] / np.array([camera_intrinsic_matrix[0, 0], camera_intrinsic_matrix[1, 1]]))
                    vertices_3d_cam_baked[:, 2] = z_vals

                    # rendering
                    vis_image = render_mesh(
                        vis_image,
                        vertices_3d_cam_baked,
                        smpl_model.faces,
                        {
                            'focal': [camera_intrinsic_matrix[0, 0], camera_intrinsic_matrix[1, 1]],
                            'princpt': [camera_intrinsic_matrix[0, 2], camera_intrinsic_matrix[1, 2]]
                        },
                        alpha=0.8
                    )

                    # Optionally visualize the vertices to check the mesh projection
                    # vis_image = vis_mesh(vis_image,
                    #                      project_points_cam_to_2d(joints3d_b[pid].cpu().numpy(), camera_intrinsic_matrix,
                    #                                               distortion_coeffs),
                    #                      alpha=0.8,
                    #                      color_range=(0.2, 0.4),
                    #                      )

                smpl_joints_3d = joints3d_b[0].cpu().numpy()  # (24,3) already in camera coordinates
                pred_op_joints_3d, mask = smpl_to_openpose(smpl_joints_3d)  # (25,3)

                gt_joints_3d = np.array(annotations_batch[b][1]["joints"])[:, :3]
                gt_joints_3d_cam = transform_points_to_camera(gt_joints_3d, camera_extrinsic_matrix)
                # midHip = Index 8 in BODY_25 -- to compute MPJPE we need to first align the joints by the pelvis (midHip)
                gt_midHip = gt_joints_3d_cam[8].copy()  # (3,)
                pred_midHip = pred_op_joints_3d[8].copy()  # (3,)

                gt_aligned = gt_joints_3d_cam - gt_midHip
                pred_aligned = pred_op_joints_3d - pred_midHip

                gt_joints_prob = np.array(annotations_batch[b][1]["joints"])[:, 3]
                errors_mask = np.argwhere((gt_joints_prob * mask) > 0)
                gt_joints_3d_cam = gt_joints_3d_cam[errors_mask].reshape(-1, 3)
                pred_op_joints_3d = pred_op_joints_3d[errors_mask].reshape(-1, 3)
                gt_aligned = gt_aligned[errors_mask].reshape(-1, 3)
                pred_aligned = pred_aligned[errors_mask].reshape(-1, 3)
                errors = np.linalg.norm(gt_aligned - pred_aligned, axis=1)
                mpjpe_3d_mm = np.mean(errors)
                mpjpe_values[frame_idx] = mpjpe_3d_mm
                cv2.putText(vis_image,
                            f"MPJPE: {mpjpe_3d_mm:.1f} mm",
                            text_pos,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0,
                            color=(0, 0, 255),
                            thickness=2)

                gt_joints_2d = project_points_cam_to_2d(gt_joints_3d_cam, camera_intrinsic_matrix, distortion_coeffs)
                pred_joints_2d = project_points_cam_to_2d(pred_op_joints_3d, camera_intrinsic_matrix, distortion_coeffs)

                for i in range(gt_joints_2d.shape[0]):
                    x_gt, y_gt = int(gt_joints_2d[i, 0]), int(gt_joints_2d[i, 1])
                    cv2.circle(vis_image, (x_gt, y_gt), 4, (255, 0, 0), -1)

                    x_pred, y_pred = int(pred_joints_2d[i, 0]), int(pred_joints_2d[i, 1])
                    cv2.circle(vis_image, (x_pred, y_pred), 4, (0, 255, 0), -1)

                    cv2.line(vis_image, (x_gt, y_gt), (x_pred, y_pred), (255, 255, 0), 2)

                output_path = osp.join(output_frames_folder, f"{frame_idx:06d}.jpg")
                cv2.imwrite(output_path, vis_image)
                print(f"[Frame {frame_idx}] MPJPE = {mpjpe_3d_mm:.1f} mm -> Saved {output_path}")
        except:
            for b, image in enumerate(bgr_images_batch):
                cv2.putText(image, "Error during inference", (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                            color=(0, 0, 255), thickness=2)
                frame_idx = annotations_batch[b][0]
                cv2.imwrite(osp.join(output_frames_folder, f'{frame_idx:06d}.jpg'), image)

    cap.release()

    # save eval results to csv file
    output_csv_file = osp.join(output_frames_folder, "mpjpe_results.csv")
    save_mpjpe_to_csv(mpjpe_values, output_csv_file)

    # save frames to video
    output_video_path = osp.join(output_video_folder, "output_video.avi")
    frames_to_video(output_frames_folder, output_video_path, video_timestamps)

    print("Done.")


if __name__ == "__main__":
    main()
