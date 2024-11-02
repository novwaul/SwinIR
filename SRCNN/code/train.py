import time
import torch
from utils import getPSNR, getSSIM, RGB2Y

class TrainNet():
    def __init__(self, train_dataloader, valid_dataloader, device, net, rival_net, preprocess, criterion, optimizer, scheduler=None, writer=None):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.net = net
        self.bicubic = rival_net
        self.preprocess = preprocess
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
    
    def train(self, max_epoch, check_pnt_path):
        best_psnr_diff = -100.0
        for epoch in range(max_epoch):

            train_ts = time.time()
            ts = time.time()

            self.net.train()
            for iteration, (train_img, train_lbl) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad()

                img = train_img.to(self.device)
                lbl = train_lbl.to(self.device) 
                out = self.net(self.preprocess(img))
                loss = self.criterion(out, lbl)
                loss.backward()
                
                self.optimizer.step()

                if iteration%12 == 0:

                    out_cpu = out.detach().to('cpu')
                    img_cpu = self.bicubic(img.detach()).to('cpu')
                    lbl_cpu = lbl.detach().to('cpu')

                    if self.writer and epoch%10 == 9 or epoch == 0:
                        self.writer.add_images(tag='training upscale/label', img_tensor=lbl_cpu, global_step=epoch+1)
                        self.writer.add_images(tag='training upscale/srcnn', img_tensor=out_cpu, global_step=epoch+1)
                        self.writer.add_images(tag='training upscale/bicubic', img_tensor=img_cpu, global_step=epoch+1)

                    out_y_np = RGB2Y(out_cpu.numpy())
                    img_y_np = RGB2Y(img_cpu.numpy())
                    lbl_y_np = RGB2Y(lbl_cpu.numpy())

                    psnr = getPSNR(out_y_np, lbl_y_np)
                    ssim = getSSIM(out_y_np, lbl_y_np)
                    bicubic_psnr = getPSNR(img_y_np, lbl_y_np)
                    bicubic_ssim = getSSIM(img_y_np, lbl_y_np)

                    te = time.time()

                    if self.writer:
                        self.writer.add_scalar('training loss', loss.item(), epoch*len(self.train_dataloader) + iteration+1)
                        self.writer.add_scalars('training psnr', {'srcnn psnr': psnr, 'bicubic psnr': bicubic_psnr}, epoch*len(self.train_dataloader) + iteration+1)
                        self.writer.add_scalars('training ssim', {'srcnn ssim': ssim, 'bicubic ssim': bicubic_ssim}, epoch*len(self.train_dataloader) + iteration+1)
                        print(f'Epoch: {epoch+1}/100 | {iteration+1}/{len(self.train_dataloader)} | loss: {loss.item():.3f} | psnr: {psnr:.3f} | ssim: {ssim:.3f} | elapsed time: {te-ts:.3f}')
                    ts = te
            
            if self.scheduler:
                self.scheduler.step()
            
            train_te = time.time()

            print(f'Epoch: {epoch+1}/100 ends | elapsed time: {train_te-train_ts:.3f}')

            self.net.eval()
            with torch.no_grad():
                valid_ts = time.time()
                lossSum = 0
                psnrSum = 0
                ssimSum = 0
                bicubic_psnrSum = 0
                bicubic_ssimSum = 0
                for idx, (valid_img, valid_lbl) in enumerate(self.valid_dataloader):
                    img = valid_img.to(self.device)
                    lbl = valid_lbl.to(self.device)
                    out = self.net(self.preprocess(img))
                    loss = self.criterion(out, lbl)

                    out_cpu = out.to('cpu')
                    img_cpu = self.bicubic(img).to('cpu')
                    lbl_cpu = lbl.to('cpu')
                    if self.writer and (epoch%10 == 9 or epoch == 0) and idx == 0:
                        self.writer.add_images(tag='validation upscale/label', img_tensor=lbl_cpu, global_step=epoch+1)
                        self.writer.add_images(tag='validation upscale/srcnn', img_tensor=out_cpu, global_step=epoch+1)
                        self.writer.add_images(tag='validation upscale/bicubic', img_tensor=img_cpu, global_step=epoch+1)

                    out_y_np = RGB2Y(out_cpu.numpy())
                    img_y_np = RGB2Y(img_cpu.numpy())
                    lbl_y_np = RGB2Y(lbl_cpu.numpy())

                    psnr = getPSNR(out_y_np, lbl_y_np)
                    ssim = getSSIM(out_y_np, lbl_y_np)
                    bicubic_psnr = getPSNR(img_y_np, lbl_y_np)
                    bicubic_ssim = getSSIM(img_y_np, lbl_y_np)

                    lossSum += loss.item()
                    psnrSum += psnr
                    ssimSum += ssim
                    bicubic_psnrSum += bicubic_psnr
                    bicubic_ssimSum += bicubic_ssim
                
                valid_te = time.time()

                num = len(self.valid_dataloader)
                if self.writer:
                    self.writer.add_scalar('validation mean loss', lossSum/num, epoch+1)
                    self.writer.add_scalars('validation mean psnr', {'srcnn mean psnr': psnrSum/num, 'bicubic mean psnr': bicubic_psnrSum/num}, epoch+1)
                    self.writer.add_scalars('validation mean ssim', {'srcnn mean ssim': ssimSum/num, 'bicubic mean ssim': bicubic_ssimSum/num}, epoch+1)
                
                print(f'Epoch: {epoch+1}/100 | val avg loss: {lossSum/num:.3f} | val avg psnr: {psnrSum/num:.3f} | val avg ssim: {ssimSum/num:.3f} | elapsed time: {valid_te-valid_ts:.3f}')

                diff = (psnrSum-bicubic_psnrSum)/num
                if diff > best_psnr_diff:
                    best_psnr_diff = diff
                    torch.save(self.net.state_dict(), check_pnt_path)