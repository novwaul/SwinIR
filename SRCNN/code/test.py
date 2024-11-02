import time
import torch
from utils import getPSNR, getSSIM, RGB2Y

class TestNet():
    def __init__(self, check_pnt_path, device, net, rival_net, preprocess, criterion, writer=None):
        self.device = device
        self.net = net 
        self.bicubic = rival_net
        self.preprocess = preprocess
        self.criterion = criterion
        self.writer = writer
        self.net.load_state_dict(torch.load(check_pnt_path))

    def test(self, test_dataloader, data_name):
        self.test_dataloader = test_dataloader
        self.net.eval()
        with torch.no_grad():
            test_ts = time.time()
            psnrSum = 0
            ssimSum = 0
            bicubic_psnrSum = 0
            bicubic_ssimSum = 0
            for idx, (test_img, test_lbl) in enumerate(self.test_dataloader):
                img = test_img.to(self.device)
                lbl = test_lbl.to(self.device)
                out = self.net(self.preprocess(img))
                loss = self.criterion(out, lbl)

                out_cpu = out.to('cpu')
                img_cpu = self.bicubic(img).to('cpu')
                lbl_cpu = lbl.to('cpu')
                
                if self.writer:
                    self.writer.add_images(tag='test upscale/'+data_name+'/label', img_tensor=lbl_cpu, global_step=idx+1)
                    self.writer.add_images(tag='test upscale/'+data_name+'/srcnn', img_tensor=out_cpu, global_step=idx+1)
                    self.writer.add_images(tag='test upscale/'+data_name+'/bicubic', img_tensor=img_cpu, global_step=idx+1)

                out_y_np = RGB2Y(out_cpu.numpy())
                img_y_np = RGB2Y(img_cpu.numpy())
                lbl_y_np = RGB2Y(lbl_cpu.numpy())

                psnr = getPSNR(out_y_np, lbl_y_np)
                ssim = getSSIM(out_y_np, lbl_y_np)
                bicubic_psnr = getPSNR(img_y_np, lbl_y_np)
                bicubic_ssim = getSSIM(img_y_np, lbl_y_np)

                psnrSum += psnr
                ssimSum += ssim
                bicubic_psnrSum += bicubic_psnr
                bicubic_ssimSum += bicubic_ssim
            
            test_te = time.time()

            num = len(self.test_dataloader)
            if self.writer:
                self.writer.add_scalars('test mean psnr - '+data_name, {'srcnn mean psnr': psnrSum/num, 'bicubic mean psnr': bicubic_psnrSum/num})
                self.writer.add_scalars('test mean ssim - '+data_name, {'srcnn mean ssim': ssimSum/num, 'bicubic mean ssim': bicubic_ssimSum/num})
            
            print(f'test avg psnr: {psnrSum/num:.3f} | test avg ssim: {ssimSum/num:.3f} | bicubic avg psnr: {bicubic_psnrSum/num:.3f} | bicubic avg ssim: {bicubic_ssimSum/num:.3f} | elapsed time: {test_te-test_ts:.3f}')
