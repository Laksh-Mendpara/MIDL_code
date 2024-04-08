import torch
import numpy as np
import copy
import os
import time
import torch.nn as nn
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

class YourModelTrainer:
    def __init__(self, sr3, dataloader, testloader, optimizer, save_path, device, img_size, LR_size):
        self.sr3 = sr3
        self.dataloader = dataloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.img_size = img_size
        self.LR_size = LR_size

    def train(self, epoch, verbose):
        fixed_imgs1 = copy.deepcopy(next(iter(self.testloader)))
        fixed_imgs1 = fixed_imgs1[0].to(self.device)
        fixed_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(fixed_imgs1))
        train_losses = []
        val_losses = []
        ssim_values = []  
        psnr_values = []  
        training_starttime = time.time()

        for i in range(epoch):
            start_time = time.time()
            train_loss = 0
            KLD_loss = 0
            kl_loss = 0
            recon_loss = 0
            for _, imgs in enumerate(self.dataloader):
                boxes = imgs[1].to(self.device)
                imgs = imgs[0].to(self.device)
                b, c, h, w = imgs.shape

                x_min = boxes[0, 0, 0].item()
                y_min = boxes[0, 0, 1].item()
                x_max = boxes[0, 0, 2].item()
                y_max = boxes[0, 0, 3].item()

                x_min = int(x_min)
                x_max = int(x_max)
                y_min = int(y_min)
                y_max = int(y_max)

                self.optimizer.zero_grad()

                if (x_max > x_min and y_max > y_min and
                        x_max <= w and x_min >= 0 and
                        y_max <= h and y_min >= 0):
                    roi_imgs = imgs[:, :, x_min:x_max, y_min:y_max]
                    roi_loss = self.sr3(roi_imgs)
                    roi_loss = roi_loss.sum() / int(b * c * h * w)
                else:
                    roi_loss = 0

                pred_SR = self.sr3(imgs)
                loss = pred_SR.sum() / int(b * c * h * w)

                total_loss = loss + 0.2 * roi_loss if (x_max > x_min and y_max > y_min and
                                                        x_max <= w and x_min >= 0 and
                                                        y_max <= h and y_min >= 0) else loss
                total_loss += KLD_loss + recon_loss  

                total_loss.backward()
                self.optimizer.step()
                train_loss += total_loss.item() * b
                kl_loss += KLD_loss.item() * b
                recon_loss += recon_loss.item() * b

            self.sr3.eval()
            test_imgs = next(iter(self.testloader))
            test_imgs = test_imgs[0].to(self.device)
            b, c, h, w = test_imgs.shape

            with torch.no_grad():
                val_loss = self.sr3(test_imgs)
                val_loss = val_loss.sum() / int(b * c * h * w)

                high_resolution_generated = self.sr3(fixed_imgs1).detach().cpu().numpy()
                original_img = fixed_imgs1[0].permute(1, 2, 0).cpu().numpy()
                generated_img = high_resolution_generated.squeeze().transpose(1, 2, 0)
                ssim_value = ssim(original_img, generated_img, multichannel=True)
                psnr_value = calculate_psnr(original_img, generated_img)
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)

                if verbose:
                    print(f'Epoch: {i + 1} / SSIM:{ssim_value:.3f} / PSNR:{psnr_value:.3f}')

            self.sr3.train()

            train_loss = train_loss / len(self.dataloader)
            kl_loss = kl_loss / len(self.dataloader)
            recon_loss = recon_loss / len(self.dataloader)
            train_losses.append(train_loss)
            val_losses.append(val_loss.item())

            os.makedirs("/home/dattatreyo/sr3_try/photo_output/originalnew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/photo_output/lownew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/photo_output/highnew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/loss_plots", exist_ok=True)

            self.save(self.save_path)
            plot_losses(train_losses, val_losses, i + 1)
            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            if verbose:
                print("Execution Time: ", format(round(execution_time_minutes, 2)), "minutes")

        if verbose:
            print("\n")
        plot_losses(train_losses, val_losses, epoch)
        training_endtime = time.time()
        training_execution_time = training_endtime - training_starttime
        training_execution_time_minutes = training_execution_time / 60
        if verbose:
            print("Training Execution Time:", format(round(training_execution_time_minutes, 2)), "minutes")

    def test(self, imgs):
        imgs_lr = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(imgs))
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs_lr)
            else:
                result_SR = self.sr3.super_resolution(imgs_lr)
            
            boxes = imgs[1].to(self.device)
            imgs = imgs[0].to(self.device)
            b, c, h, w = imgs.shape

            x_min = boxes[0, 0, 0].item()
            y_min = boxes[0, 0, 1].item()
            x_max = boxes[0, 0, 2].item()
            y_max = boxes[0, 0, 3].item()

            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)

            if (x_max > x_min and y_max > y_min and
                    x_max <= w and x_min >= 0 and
                    y_max <= h and y_min >= 0):
                roi_imgs = imgs[:, :, x_min:x_max, y_min:y_max]
                roi_result_SR = self.sr3(roi_imgs)
            else:
                roi_result_SR = None

        self.sr3.train()
        return result_SR, roi_result_SR, imgs

    def evaluate_performance(self, runs=10):
        psnr_values = []
        ssim_values = []

        for _ in range(runs):
            sr_outputs = []
            original_images = []
            for imgs in self.testloader:
                sr_output, _, original_img = self.test(imgs)
                sr_outputs.append(sr_output.cpu().detach().numpy())
                original_images.append(original_img.cpu().detach().numpy())
            
            psnr_per_run = []
            ssim_per_run = []
            for sr_output, original_img in zip(sr_outputs, original_images):
                psnr_per_image = calculate_psnr(sr_output, original_img)
                ssim_per_image = ssim(sr_output, original_img, multichannel=True)
                psnr_per_run.append(psnr_per_image)
                ssim_per_run.append(ssim_per_image)
            
            psnr_values.append(np.mean(psnr_per_run))
            ssim_values.append(np.mean(ssim_per_run))

        mean_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        mean_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)

        print("Mean PSNR:", mean_psnr)
        print("Standard Deviation PSNR:", std_psnr)
        print("Mean SSIM:", mean_ssim)
        print("Standard Deviation SSIM:", std_ssim)
