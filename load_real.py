import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F


# fmt: off
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()
# fmt: on


def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    ct = torch.Tensor([[1, 0, 0, wx], [0, 1, 0, wy], [0, 0, 1, wz], [0, 0, 0, 1]]).float()
    c2w = ct @ c2w

    return c2w


def load_real_capture_frame_data(basedir, half_res=False):
    # frame data
    all_imgs = []
    all_poses = []
    all_hwf = []
    all_time_steps = []
    counts = [0]
    merge_counts = [0]
    t_info = [0.0, 0.0, 0.0, 0.0]


    with open(os.path.join(basedir, "transforms_aligned.json"), "r") as fp:
        # read render settings
        meta = json.load(fp)
    near = float(meta["near"])
    far = float(meta["far"])
    radius = (near + far) * 0.5
    phi = 20.0
    rotZ = False
    r_center = np.array([0.3382070094283088, 0.38795384153014023, -0.2609209839653898]).astype(np.float32)

    # read scene data
    voxel_tran = np.array(
        [
            [0.0, 0.0, 1.0, 0.081816665828228],
            [0.0, 1.0, 0.0, -0.044627271592617035],
            [-1.0, 0.0, 0.0, -0.004908999893814325],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # swap_zx
    voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]], axis=1)
    voxel_scale = np.broadcast_to([0.4909, 0.73635, 0.4909], [3])

    # read video frames
    # all videos should be synchronized, having the same frame_rate and frame_num

    frames = meta["frames"]

    target_cam_names_dict = {"train": ["2"], "val": ["2"], "test": ["2"]}
    frame_nums = 120
    if "red" in basedir.lower():
        print("red")
        start_i = 33
    elif "blue" in basedir.lower():
        print("blue")
        start_i = 55
    else:
        raise ValueError("Unknown dataset")

    delta_t = 1.0 / frame_nums

    for s in ["train", "val", "test"]:
        target_cam_names = target_cam_names_dict[s]
        for frame_dict in frames:
            cam_name = frame_dict["file_path"][-1:]  # train0x -> x used to determine with train_views
            if cam_name not in target_cam_names:
                continue
            print(f"cam_name: {cam_name}")

            camera_angle_x = float(frame_dict["camera_angle_x"])
            camera_hw = frame_dict["camera_hw"]
            H, W = camera_hw

            Focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
            if half_res != "normal":
                if half_res == "half":  # errors if H or W is not dividable by 2
                    H = H // 2
                    W = W // 2
                    Focal = Focal / 2.0
                elif half_res == "quater":  # errors if H or W is not dividable by 4
                    H = H // 4
                    W = W // 4
                    Focal = Focal / 4.0
                elif half_res == "double":
                    H = H * 2
                    W = W * 2
                    focal = focal * 2.0

            imgs = []
            poses = []
            time_steps = []
            pose = np.array(frame_dict["transform_matrix"]).astype(np.float32)

            for time_idx in range(start_i, start_i + frame_nums * 2, 2):

                frame_path = os.path.join(frame_dict["file_path"], f"{time_idx:03d}.png")
                frame = cv2.imread(os.path.join(basedir, frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if half_res:
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

                imgs.append(frame)
                time_step = time_idx - start_i
                time_step /= 2
                time_steps.append([time_step * delta_t])

            # print(f"video {train_video['file_name']} focal {Focal}")
            imgs = np.float32(imgs) / 255.0
            poses = np.array([pose] * len(imgs)).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_time_steps.append(time_steps)
            all_poses.append(poses)
            all_hwf.append(np.float32([[H, W, Focal]] * imgs.shape[0]))

        merge_counts.append(counts[-1])

    t_info = np.float32([0.0, 1.0, 0.5, delta_t])  # min t, max t, mean t, delta_t
    i_split = [np.arange(merge_counts[i], merge_counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    time_steps = np.concatenate(all_time_steps, 0)
    hwfs = np.concatenate(all_hwf, 0)

    # set render settings:
    sp_n = 40  # an even number!
    sp_poses = [
        pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2])
        for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
    ]
    render_poses = torch.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps = np.arange(sp_n) / (sp_n - 1)
    bkg_color = np.array([0.0, 0.0, 0.0])

    return (
        imgs,
        poses,
        time_steps,
        hwfs,
        render_poses,
        render_timesteps,
        i_split,
        t_info,
        voxel_tran,
        voxel_scale,
        bkg_color,
        near,
        far,
    )
