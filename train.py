import torch
import time
import os
import copy
from skimage.metrics import structural_similarity as ssim
from utils import calculate_psnr, plot_losses

class sr3:
    def __init__(self, model, dataloader, testloader, optimizer, device, epoch, save_path, LR_size, img_size):
        self.model = model
        self.dataloader = dataloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.device = device
        self.epoch = epoch
        self.save_path = save_path
        self.LR_size = LR_size
        self.img_size = img_size

    def train(self, verbose):
        fixed_imgs1 = copy.deepcopy(next(iter(self.testloader)))
        fixed_imgs1 = fixed_imgs1[0].to(self.device)
        fixed_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(fixed_imgs1))
        train_losses = []
        val_losses = []
        ssim_values = []
        psnr_values = []
        training_starttime = time.time()

        for i in range(self.epoch):
            start_time = time.time()
            train_loss = 0
            KLD_loss = 0
            kl_loss = 0
            recon_loss = 0

            for _, imgs in enumerate(self.dataloader):
                boxes = imgs[1].to(self.device)
                imgs = imgs[0].to(self.device)

                self.optimizer.zero_grad()

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
                    roi_loss = self.model(roi_imgs)
                    roi_loss = roi_loss.sum() / int(b * c * h * w)
                else:
                    roi_loss = 0

                pred_SR, KLD_loss, recon_loss = self.model(imgs)
                loss = pred_SR.sum() / int(b * c * h * w)

                total_loss = 0.4 * loss + 0.3 * (KLD_loss + recon_loss) + 0.3 * roi_loss if (x_max > x_min and y_max > y_min and
                                                        x_max <= w and x_min >= 0 and
                                                        y_max <= h and y_min >= 0) else loss
                
                total_loss.backward()
                self.optimizer.step()
                train_loss += total_loss.item() * b
                kl_loss += KLD_loss.item() * b
                recon_loss += recon_loss.item() * b

            self.model.eval()
            test_imgs = next(iter(self.testloader))
            test_imgs = test_imgs[0].to(self.device)
            b, c, h, w = test_imgs.shape

            with torch.no_grad():
                val_loss, KLD_loss, recon_loss = self.model(test_imgs)
                val_loss = val_loss.sum() / int(b * c * h * w)

                high_resolution_generated = self.model(fixed_imgs1).detach().cpu().numpy()
                original_img = fixed_imgs1[0].permute(1, 2, 0).cpu().numpy()
                generated_img = high_resolution_generated.squeeze().transpose(1, 2, 0)
                ssim_value = ssim(original_img, generated_img, multichannel=True)
                psnr_value = calculate_psnr(original_img, generated_img)
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)

                if verbose:
                    print(f'Epoch: {i + 1} / SSIM:{ssim_value:.3f} / PSNR:{psnr_value:.3f}')

            self.model.train()

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
        plot_losses(train_losses, val_losses, self.epoch)
        training_endtime = time.time()
        training_execution_time = training_endtime - training_starttime
        training_execution_time_minutes = training_execution_time / 60
        if verbose:
            print("Training Execution Time:", format(round(training_execution_time_minutes, 2)), "minutes")
