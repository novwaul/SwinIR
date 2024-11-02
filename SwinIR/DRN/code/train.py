import torch
import torch.nn as nn
from time import time
from utils import calc_psnr, calc_ssim, cvrt_rgb_to_y, norm, denorm
    
def train(args, resume):

    net = args['net']
    bicubic = args['bicubic']

    optimizer = args['optimizer']
    scheduler = args['scheduler']

    criterion = args['criterion']

    device = args['device']
    epochs = args['epochs']
    crop_out = args['crop_out']

    train_dataloaders = args['train_dataloaders']
    valid_dataloader = args['valid_dataloader']

    check_pnt_path = args['check_pnt_path']
    last_pnt_path = args['last_pnt_path']
    old_pnt_path = args['old_pnt_path']

    writer = args['writer']

    downsample_2x = nn.Upsample(scale_factor=0.5, mode='bicubic').to(device)
    downsample_1x = nn.Upsample(scale_factor=0.25, mode='bicubic').to(device)

    if resume:
        states = torch.load(last_pnt_path)
        net.load_state_dict(states['net'])
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])
        best_psnr_diff = states['best_psnr_diff']
        best_ssim_diff = states['best_ssim_diff']
        epoch = states['epoch']
        total_time = states['total_time']
    else:
        best_ssim_diff = -100.0
        best_psnr_diff = -100.0
        epoch = 0
        total_time = 0.0

    total_iterations = sum([len(train_dataloader) for train_dataloader in train_dataloaders])
    batch_num = len(valid_dataloader)

    while epoch < epochs:

        start = time()

        step = epoch*total_iterations
        
        net.train()

        for i, train_dataloader in enumerate(train_dataloaders):
            for j, train_images in enumerate(train_dataloader):
            
                img, lbl_2x, lbl_4x = train_images

                lr = norm(bicubic(img.to(device)))
                lbl_1x = norm(img.to(device))
                lbl_2x = norm(lbl_2x.to(device))
                lbl_4x = norm(lbl_4x.to(device))
                    
                iteration = j + sum([len(train_dataloaders[k]) for k in range(0, i)])
                
                optimizer.zero_grad()

                hr, p_1x, p_2x, d_1x, d_2x = net(lr)

                p_loss_4x = criterion(hr, lbl_4x)
                p_loss_2x = criterion(p_2x, lbl_2x)
                p_loss_1x = criterion(p_1x, lbl_1x)
                d_loss_2x = 0.1*criterion(d_2x, lbl_2x)
                d_loss_1x = 0.1*criterion(d_1x, lbl_1x)

                loss = p_loss_4x + p_loss_2x + p_loss_1x + d_loss_2x + d_loss_1x
                loss.backward()
                
                optimizer.step()

                if iteration%300 == 299:
                    hr_cpu = denorm(hr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                    lr_cpu = denorm(lr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                    lbl_4x_cpu = denorm(lbl_4x.detach()).to('cpu')

                    hr_y_np = cvrt_rgb_to_y(hr_cpu.numpy())
                    lr_y_np = cvrt_rgb_to_y(lr_cpu.numpy())
                    lbl_4x_y_np = cvrt_rgb_to_y(lbl_4x_cpu.numpy())

                    psnr = calc_psnr(hr_y_np, lbl_4x_y_np, crop_out)
                    ssim = calc_ssim(hr_y_np, lbl_4x_y_np, crop_out)
                    bicubic_psnr = calc_psnr(lr_y_np, lbl_4x_y_np, crop_out)
                    bicubic_ssim = calc_ssim(lr_y_np, lbl_4x_y_np, crop_out)

                    writer.add_scalar('Train Loss/A. Total', loss.item(),  step+iteration)
                    writer.add_scalars('Train Loss/B. Primal', {'4x': p_loss_4x.item(), '2x': p_loss_2x.item(), '1x': p_loss_1x.item()}, step+iteration)
                    writer.add_scalars('Train Loss/C. Dual', {'2x': d_loss_2x.item(), '1x': d_loss_1x.item()}, step+iteration)
                    writer.add_scalars('Train PSNR', {'Model PSNR': psnr, 'Bicubic PSNR': bicubic_psnr}, step+iteration)
                    writer.add_scalars('Train SSIM', {'Model SSIM': ssim, 'Bicubic SSIM': bicubic_ssim}, step+iteration)
                    print(f'Epoch: {epoch+1}/{epochs} | {iteration+1}/{total_iterations} | Loss: {loss.item():.3f} | PSNR: {psnr:.3f} | SSIM: {ssim:.3f}')
                    
        scheduler.step() 

        net.eval()

        with torch.no_grad():

            total_loss = 0.0
            total_p_loss_4x = 0.0
            total_p_loss_2x = 0.0
            total_p_loss_1x = 0.0
            total_d_loss_2x = 0.0
            total_d_loss_1x = 0.0

            total_psnr = 0.0
            total_ssim = 0.0
            total_bicubic_psnr = 0.0
            total_bicubic_ssim = 0.0

            for iteration, (img, lbl_2x, lbl_4x) in enumerate(valid_dataloader):

                lbl_1x = norm(img.to(device))
                lbl_2x = norm(lbl_2x.to(device))
                lbl_4x = norm(lbl_4x.to(device))

                lr = norm(bicubic(img.to(device)))

                hr, p_1x, p_2x, d_1x, d_2x = net(lr)

                p_loss_4x = criterion(hr, lbl_4x)
                p_loss_2x = criterion(p_2x, lbl_2x)
                p_loss_1x = criterion(p_1x, lbl_1x)
                d_loss_2x = 0.1*criterion(d_2x, lbl_2x)
                d_loss_1x = 0.1*criterion(d_1x, lbl_1x)

                loss = p_loss_4x + p_loss_2x + p_loss_1x + d_loss_2x + d_loss_1x

                hr_cpu = denorm(hr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                lr_cpu = denorm(lr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                lbl_4x_cpu = denorm(lbl_4x.detach()).to('cpu')
                
                if iteration == 0 and (epoch%5 == 4 or epoch == 0):
                    writer.add_images(tag='Valid Upscale/A. Ground Truth', img_tensor=lbl_4x_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/B. Bicubic', img_tensor=lr_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/C. Model', img_tensor=hr_cpu, global_step=epoch+1)

                hr_y_np = cvrt_rgb_to_y(hr_cpu.numpy())
                lr_y_np = cvrt_rgb_to_y(lr_cpu.numpy())
                lbl_4x_y_np = cvrt_rgb_to_y(lbl_4x_cpu.numpy())

                psnr = calc_psnr(hr_y_np, lbl_4x_y_np, crop_out)
                ssim = calc_ssim(hr_y_np, lbl_4x_y_np, crop_out)
                bicubic_psnr = calc_psnr(lr_y_np, lbl_4x_y_np, crop_out)
                bicubic_ssim = calc_ssim(lr_y_np, lbl_4x_y_np, crop_out)

                total_loss += loss.item()
                total_p_loss_4x += p_loss_4x.item()
                total_p_loss_2x += p_loss_2x.item()
                total_p_loss_1x += p_loss_1x.item()
                total_d_loss_2x += d_loss_2x.item()
                total_d_loss_1x += d_loss_1x.item()

                total_psnr += psnr
                total_ssim += ssim
                total_bicubic_psnr += bicubic_psnr
                total_bicubic_ssim += bicubic_ssim

            avg_total_loss = total_loss/batch_num
            avg_p_loss_4x = total_p_loss_4x/batch_num
            avg_p_loss_2x = total_p_loss_2x/batch_num
            avg_p_loss_1x = total_p_loss_1x/batch_num
            avg_d_loss_2x = total_d_loss_2x/batch_num
            avg_d_loss_1x = total_d_loss_1x/batch_num

            avg_psnr = total_psnr/batch_num
            avg_ssim = total_ssim/batch_num
            avg_bicubic_psnr = total_bicubic_psnr/batch_num
            avg_bicubic_ssim = total_bicubic_ssim/batch_num
            
            writer.add_scalar('Valid Loss/A. Total', avg_total_loss,  step)
            writer.add_scalars('Valid Loss/B. Primal', {'4x': avg_p_loss_4x, '2x': avg_p_loss_2x, '1x': avg_p_loss_1x}, step)
            writer.add_scalars('Valid Loss/C. Dual', {'2x': avg_d_loss_2x, '1x': avg_d_loss_1x}, step)
            writer.add_scalars('Valid PSNR', {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr}, step)
            writer.add_scalars('Valid SSIM', {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim}, step)
            
            end = time()
            elpased_time = end - start
            print(f'Epoch: {epoch+1}/{epochs} | Time: {elpased_time:.3f} | Val Loss: {avg_total_loss:.3f} | Val PSNR: {avg_psnr:.3f} | Val SSIM: {avg_ssim:.3f}')

            total_time += elpased_time

            psnr_diff = avg_psnr - avg_bicubic_psnr
            ssim_diff = avg_ssim - avg_bicubic_ssim
            if psnr_diff > best_psnr_diff:
                best_psnr_diff = psnr_diff
                torch.save(net.state_dict(), check_pnt_path)
            elif ssim_diff > best_ssim_diff:
                best_ssim_diff = ssim_diff
                torch.save(net.state_dict(), check_pnt_path)
        
        if epoch > 0:
            old_states = torch.load(last_pnt_path)
            torch.save(old_states, old_pnt_path)
        
        epoch += 1

        states = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr_diff': best_psnr_diff,
            'best_ssim_diff': best_ssim_diff,
            'epoch': epoch,
            'total_time': total_time
        }

        torch.save(states, last_pnt_path)
