import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lpips import LPIPS
from skimage.metrics import structural_similarity


def compute_psnr(img1, img2):
    # Ensure the images are of the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is zero, the PSNR is infinite (images are identical)
    if mse == 0:
        return float("inf")

    # Use the maximum pixel value of the image as MAX_I
    print(mse)
    return -10 * np.log10(mse)


video_path = "/viscam/u/y1gao/pinf_smoke/data/ScalarReal/higher_viscosity/train02.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
lpips_net = LPIPS().cuda()
frame_count = 0
total_psnr = 0
frame_0 = None
i = 0
total_lpips = 0
total_ssim = 0
while True:
    ret, frame = cap.read()  # Read the next frame

    # print(frame.min(), frame.max())
    if not ret:
        break  # Break if the video is finished
    frame = (frame / 255).astype(np.float32)

    print(frame.min(), frame.max())
    # Create a black frame of the same shape as the video frame
    black_frame = plt.imread(
        "/viscam/u/y1gao/pinf_smoke/log/syn_higher_viscosity/run_advect_den_600000/rgb_{:03d}.png".format(frame_count)
    ).astype(np.float32)
    # print(frame.shape)
    black_frame = cv2.resize(black_frame, (black_frame.shape[1] * 2, black_frame.shape[0] * 2))[
        50 * 2 : 531 * 2, 25 * 2 : 300 * 2
    ]
    frame = frame[50 * 2 : 531 * 2, 25 * 2 : 300 * 2]
    # black_frame = black_frame[50:531,25:300]
    # frame = cv2.resize(frame, (frame.shape[1]//2,frame.shape[0]//2))[50:531,25:300]
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
    black_frame_tensor = torch.from_numpy(black_frame).permute(2, 0, 1).unsqueeze(0).cuda()
    distance = lpips_net(frame_tensor, black_frame_tensor, normalize=True).item()
    total_lpips += distance
    # Calculate PSNR for the frame
    psnr_value = compute_psnr(frame, black_frame)

    total_psnr += psnr_value
    ssim_value = structural_similarity(frame, black_frame, data_range=1.0, channel_axis=2)
    total_ssim += ssim_value
    frame_count += 1

# Calculate the average PSNR over all frames
avg_psnr = total_psnr / frame_count
avg_lpips = total_lpips / frame_count
avg_ssim = total_ssim / frame_count
print(f"Average PSNR between each frame and a black frame: {avg_psnr:.2f} dB")
print(f"Average LPIPS between each frame and a black frame: {avg_lpips:.4f}")
print(f"Average SSIM between each frame and a black frame: {avg_ssim:.4f}")
cap.release()
