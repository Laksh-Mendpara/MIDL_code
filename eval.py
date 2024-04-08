import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_metrics(original_dir, result_dir):
    original_images = sorted(os.listdir(original_dir))
    result_images = sorted(os.listdir(result_dir))

    ssim_values = []
    psnr_values = []

    for original_img, result_img in zip(original_images, result_images):
        original_path = os.path.join(original_dir, original_img)
        result_path = os.path.join(result_dir, result_img)

        original = cv2.imread(original_path)
        result = cv2.imread(result_path)

        # Convert images to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim_score, _ = ssim(original_gray, result_gray, full=True)
        ssim_values.append(ssim_score)

        # Compute PSNR
        mse = np.mean((original_gray - result_gray) ** 2)
        if mse == 0:
            psnr = 100
        else:
            max_pixel = np.max(original_gray)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        psnr_values.append(psnr)

    ssim_mean = np.mean(ssim_values)
    ssim_sd = np.std(ssim_values)
    psnr_mean = np.mean(psnr_values)
    psnr_sd = np.std(psnr_values)

    return ssim_mean, ssim_sd, psnr_mean, psnr_sd

def main():
    original_hr_dir = "/home/dattatreyo/SHOW RESULTS/set1/original_HR"
    proposed_dir = "/home/dattatreyo/SHOW RESULTS/set1/proposed"
    sr_img_dir = "/home/dattatreyo/SHOW RESULTS/set1/SR_img"

    # Calculate metrics between original HR and proposed images
    ssim_mean_proposed, ssim_sd_proposed, psnr_mean_proposed, psnr_sd_proposed = calculate_metrics(original_hr_dir, proposed_dir)

    # Calculate metrics between original HR and SR_img images
    ssim_mean_sr_img, ssim_sd_sr_img, psnr_mean_sr_img, psnr_sd_sr_img = calculate_metrics(original_hr_dir, sr_img_dir)

    print("Mean +/- SD values between original HR and proposed:")
    print(f"SSIM: {ssim_mean_proposed:.4f} +/- {ssim_sd_proposed:.4f}")
    print(f"PSNR: {psnr_mean_proposed:.2f} dB +/- {psnr_sd_proposed:.2f}")

    print("\nMean +/- SD values between original HR and SR_img:")
    print(f"SSIM: {ssim_mean_sr_img:.4f} +/- {ssim_sd_sr_img:.4f}")
    print(f"PSNR: {psnr_mean_sr_img:.2f} dB +/- {psnr_sd_sr_img:.2f}")

if __name__ == "__main__":
    main()
