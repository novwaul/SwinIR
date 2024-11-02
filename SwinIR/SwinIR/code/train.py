import torch
import torch.nn as nn
from time import time
from utils import calc_psnr, calc_ssim, cvrt_rgb_to_y, norm, denorm
import torch.distributed as dist
    
def train(args, resume):

    net = args['net']

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

    if resume:

        dist.barrier()

        fromTo = {'cuda:%d'%0: 'cuda:%d'%dist.get_rank()}
        states = torch.load(last_pnt_path, map_location=fromTo)

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
    bicubic = nn.Upsample(scale_factor=4, mode='bicubic').to(device)
    master = (dist.get_rank() == 0)

    while epoch < epochs:

        start = time()

        step = epoch*total_iterations
        
        net.train()

        for i, train_dataloader in enumerate(train_dataloaders):
            for j, train_images in enumerate(train_dataloader):
            
                img, lbl = train_images

                lr = norm(img.to(device))
                lbl = norm(lbl.to(device))

                iteration = j + sum([len(train_dataloaders[k]) for k in range(0, i)])
                
                optimizer.zero_grad()

                hr = net(lr)

                loss = criterion(hr, lbl)

                loss.backward()
                
                optimizer.step()

                if iteration%100 == 99:
                    
                    hr_cpu = denorm(hr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                    lr_cpu = denorm(bicubic(lr).detach()).clamp(min=0.0, max=1.0).to('cpu')
                    lbl_cpu = denorm(lbl.detach()).to('cpu')

                    hr_y_np = cvrt_rgb_to_y(hr_cpu.numpy())
                    lr_y_np = cvrt_rgb_to_y(lr_cpu.numpy())
                    lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                    psnr = torch.tensor(calc_psnr(hr_y_np, lbl_y_np, crop_out)).to(device)
                    ssim = torch.tensor(calc_ssim(hr_y_np, lbl_y_np, crop_out)).to(device)
                    bicubic_psnr = torch.tensor(calc_psnr(lr_y_np, lbl_y_np, crop_out)).to(device)
                    bicubic_ssim = torch.tensor(calc_ssim(lr_y_np, lbl_y_np, crop_out)).to(device)               
                    loss_value = torch.tensor(loss.item()).to(device)

                    dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                    dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
                    dist.all_reduce(ssim, op=dist.ReduceOp.SUM)
                    dist.all_reduce(bicubic_psnr, op=dist.ReduceOp.SUM)
                    dist.all_reduce(bicubic_ssim, op=dist.ReduceOp.SUM)

                    N = dist.get_world_size()
                    avg_loss = loss_value.to('cpu').numpy()/N
                    avg_psnr = psnr.to('cpu').numpy()/N
                    avg_ssim = ssim.to('cpu').numpy()/N
                    avg_bicubic_psnr = bicubic_psnr.to('cpu').numpy()/N
                    avg_bicubic_ssim = bicubic_ssim.to('cpu').numpy()/N

                    if master:
                        writer.add_scalar('Train Loss', avg_loss,  step+iteration)
                        writer.add_scalars('Train PSNR', {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr}, step+iteration)
                        writer.add_scalars('Train SSIM', {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim}, step+iteration)
                        print(f'Epoch: {epoch+1}/{epochs} | {iteration+1}/{total_iterations} | Loss: {avg_loss:.3f} | PSNR: {avg_psnr:.3f} | SSIM: {avg_ssim:.3f}')
                        
        scheduler.step() 

        net.eval()

        with torch.no_grad():

            total_loss = 0.0
            total_psnr = 0.0
            total_ssim = 0.0
            total_bicubic_psnr = 0.0
            total_bicubic_ssim = 0.0

            for iteration, (img, lbl) in enumerate(valid_dataloader):

                lr = norm(img.to(device))
                lbl = norm(lbl.to(device))

                hr = net(lr)

                loss = criterion(hr, lbl)

                hr_cpu = denorm(hr.detach()).clamp(min=0.0, max=1.0).to('cpu')
                lr_cpu = denorm(bicubic(lr).detach()).clamp(min=0.0, max=1.0).to('cpu')
                lbl_cpu = denorm(lbl.detach()).to('cpu')
                
                if master and iteration == 5 and (epoch%20 == 19 or epoch == 0):
                    writer.add_images(tag='Valid Upscale/A. Ground Truth', img_tensor=lbl_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/B. Bicubic', img_tensor=lr_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/C. Model', img_tensor=hr_cpu, global_step=epoch+1)

                hr_y_np = cvrt_rgb_to_y(hr_cpu.numpy())
                lr_y_np = cvrt_rgb_to_y(lr_cpu.numpy())
                lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                psnr = calc_psnr(hr_y_np, lbl_y_np, crop_out)
                ssim = calc_ssim(hr_y_np, lbl_y_np, crop_out)
                bicubic_psnr = calc_psnr(lr_y_np, lbl_y_np, crop_out)
                bicubic_ssim = calc_ssim(lr_y_np, lbl_y_np, crop_out)

                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                total_bicubic_psnr += bicubic_psnr
                total_bicubic_ssim += bicubic_ssim

            loss_value = torch.tensor(total_loss/batch_num).to(device)
            psnr = torch.tensor(total_psnr/batch_num).to(device)
            ssim = torch.tensor(total_ssim/batch_num).to(device)
            bicubic_psnr = torch.tensor(total_bicubic_psnr/batch_num).to(device)
            bicubic_ssim = torch.tensor(total_bicubic_ssim/batch_num).to(device)

            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(bicubic_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(bicubic_ssim, op=dist.ReduceOp.SUM)

            N = dist.get_world_size()
            avg_loss = loss_value.to('cpu').numpy()/N
            avg_psnr = psnr.to('cpu').numpy()/N
            avg_ssim = ssim.to('cpu').numpy()/N
            avg_bicubic_psnr = bicubic_psnr.to('cpu').numpy()/N
            avg_bicubic_ssim = bicubic_ssim.to('cpu').numpy()/N

            if master:
                writer.add_scalar('Valid Loss', avg_loss,  step)
                writer.add_scalars('Valid PSNR', {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr}, step)
                writer.add_scalars('Valid SSIM', {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim}, step)
            
            end = time()
            elpased_time = end - start

            if master:
                print(f'Epoch: {epoch+1}/{epochs} | Time: {elpased_time:.3f} | Val Loss: {avg_loss:.3f} | Val PSNR: {avg_psnr:.3f} | Val SSIM: {avg_ssim:.3f}')

            total_time += elpased_time

            if master:
                psnr_diff = avg_psnr - avg_bicubic_psnr
                ssim_diff = avg_ssim - avg_bicubic_ssim

                if psnr_diff > best_psnr_diff:
                    best_psnr_diff = psnr_diff
                    torch.save(net.state_dict(), check_pnt_path)
                elif ssim_diff > best_ssim_diff:
                    best_ssim_diff = ssim_diff
                    torch.save(net.state_dict(), check_pnt_path)
        
        if master:
            if epoch > 0:
                old_states = torch.load(last_pnt_path)
                torch.save(old_states, old_pnt_path)
            
            states = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr_diff': best_psnr_diff,
                'best_ssim_diff': best_ssim_diff,
                'epoch': epoch+1,
                'total_time': total_time
            }

            torch.save(states, last_pnt_path)
        
        ### update epoch
        epoch += 1
