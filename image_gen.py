import os

import imageio
import numpy as np

from tqdm import trange


sim_path = "sim_000001"


def merge_imgs(framerate, save_dir, camera_id):
    # ffmpeg is required by torchvision but it needs some missing lib; have to use /usr/bin/ffmpeg
    os.system("ffmpeg -hide_banner -loglevel error -y -i {0}/%03d.png -vf palettegen {0}/palette.png".format(save_dir))
    os.system(
        "ffmpeg -hide_banner -loglevel error -y -framerate {framerate} -start_number {start_number} -i {save_dir}/%03d.png -i {save_dir}/palette.png -lavfi paletteuse -frames:v {n_frames} {save_dir}/output.gif".format(
            framerate=framerate, save_dir=save_dir, start_number=10, n_frames=120
        )
    )
    os.system(
        "ffmpeg -hide_banner -y -loglevel error -framerate {framerate} -start_number {start_number} -i {save_dir}/%03d.png -vf scale=1080:1920 -crf 25 -frames:v {n_frames} data/ScalarReal/{sim_path}/train0{camera_id}.mp4".format(
            framerate=framerate,
            save_dir=save_dir,
            camera_id=camera_id,
            sim_path=sim_path,
            start_number=10,
            n_frames=120,
        )
    )


def main():

    curpath = f"/viscam/u/y1gao/Global-Flow-Transport/{sim_path}"
    y_scale = 0.2 * 255
    img2cam = [2, 1, 0, 4, 3]  # ScalarFlow has a camera reordering!
    log_dirs = [os.path.join(curpath, "cam_{}".format(i)) for i in range(5)]
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
    for frame_i in trange(0, 151):
        image_path = os.path.join(curpath, "imgsTarget_%06d.npz" % frame_i)
        for cam_id in range(5):
            image_k = np.load(image_path)["data"][img2cam[cam_id]] * y_scale
            # image_k.shape = [1062, 600, 1]
            image_k = image_k.astype(np.uint8)[::-1]
            imageio.imwrite(os.path.join(log_dirs[cam_id], "{:03d}.png".format(frame_i)), image_k)
    for idx, log_dir in enumerate(log_dirs):
        merge_imgs(30, log_dir, idx)


if __name__ == "__main__":
    main()
