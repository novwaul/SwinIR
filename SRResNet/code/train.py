import torch
from time import time
from utils import calc_psnr, calc_ssim, cvrt_rgb_to_y, norm, denorm, bicubic
    
def train(args, resume):

    net = args['net']

    scale_factor = args['scale_factor']

    optimizer = args['optimizer']
    scheduler = args['scheduler']
    criterion = args['criterion']

    device = args['device']
    epochs = args['epochs']
    crop_out = args['crop_out']

    train_dataloader = args['train_dataloader']
    valid_dataloader = args['valid_dataloader']

    check_pnt_path = args['check_pnt_path']
    last_pnt_path = args['last_pnt_path']

    writer = args['writer']

    if resume:
        states = torch.load(last_pnt_path)
        net.load_state_dict(states['net'])
        scheduler.load_state_dict(states['scheduler'])
        optimizer.load_state_dict(states['optimizer'])
        best_psnr_diff = states['best_psnr_diff']
        epoch = states['epoch']
        total_time = states['total_time']
    else:
        best_psnr_diff = -100.0
        epoch = 0
        total_time = 0.0
    
    total_iterations = len(train_dataloader)
    batch_num = len(valid_dataloader)

    while epoch < epochs:

        start = time()

        step = epoch*total_iterations
        
        net.train()

        for iteration, (img, lbl) in enumerate(train_dataloader):

            optimizer.zero_grad()

            img = norm(img.to(device))
            lbl = norm(lbl.to(device))
            out = net(img)
            loss = criterion(out, lbl)
            loss.backward()
            
            optimizer.step()

            if iteration%50 == 49:

                out_cpu = denorm(out.detach()).clamp(min=0.0, max=1.0).to('cpu')
                img_cpu = bicubic(denorm(img.detach()), scale_factor).to('cpu')
                lbl_cpu = denorm(lbl.detach()).to('cpu')

                out_y_np = cvrt_rgb_to_y(out_cpu.numpy())
                img_y_np = cvrt_rgb_to_y(img_cpu.numpy())
                lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                psnr = calc_psnr(out_y_np, lbl_y_np, crop_out)
                ssim = calc_ssim(out_y_np, lbl_y_np, crop_out)
                bicubic_psnr = calc_psnr(img_y_np, lbl_y_np, crop_out)
                bicubic_ssim = calc_ssim(img_y_np, lbl_y_np, crop_out)

                writer.add_scalar('Train Loss', loss.item(),  step+iteration)
                writer.add_scalars('Train PSNR', {'Model PSNR': psnr, 'Bicubic PSNR': bicubic_psnr}, step+iteration)
                writer.add_scalars('Train SSIM', {'Model SSIM': ssim, 'Bicubic SSIM': bicubic_ssim}, step+iteration)
                print(f'Epoch: {epoch+1}/{epochs} | {iteration+1}/{len(train_dataloader)} | Loss: {loss.item():.3f} | PSNR: {psnr:.3f} | SSIM: {ssim:.3f}')
        
        scheduler.step()

        net.eval()

        with torch.no_grad():
            total_loss = 0
            total_psnr = 0
            total_ssim = 0
            total_bicubic_psnr = 0
            total_bicubic_ssim = 0
            for iteration, (img, lbl) in enumerate(valid_dataloader):

                img = norm(img.to(device))
                lbl = norm(lbl.to(device))
                out = net(img)
                loss = criterion(out, lbl)

                out_cpu = denorm(out.detach()).clamp(min=0.0, max=1.0).to('cpu')
                img_cpu = bicubic(denorm(img.detach()), scale_factor).to('cpu')
                lbl_cpu = denorm(lbl.detach()).to('cpu')
                lr_cpu = denorm(img.detach()).to('cpu')
                if iteration == 0 and (epoch%50 == 49 or epoch == 0):
                    writer.add_images(tag='Valid Upscale/Input', img_tensor=lr_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/Ground Truth', img_tensor=lbl_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/Model', img_tensor=out_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/Bicubic', img_tensor=img_cpu, global_step=epoch+1)

                out_y_np = cvrt_rgb_to_y(out_cpu.numpy())
                img_y_np = cvrt_rgb_to_y(img_cpu.numpy())
                lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                psnr = calc_psnr(out_y_np, lbl_y_np, crop_out)
                ssim = calc_ssim(out_y_np, lbl_y_np, crop_out)
                bicubic_psnr = calc_psnr(img_y_np, lbl_y_np, crop_out)
                bicubic_ssim = calc_ssim(img_y_np, lbl_y_np, crop_out)

                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                total_bicubic_psnr += bicubic_psnr
                total_bicubic_ssim += bicubic_ssim

            avg_loss = total_loss/batch_num
            avg_psnr = total_psnr/batch_num
            avg_ssim = total_ssim/batch_num
            avg_bicubic_psnr = total_bicubic_psnr/batch_num
            avg_bicubic_ssim = total_bicubic_ssim/batch_num
            
            writer.add_scalar('Valid Loss', avg_loss, step)
            writer.add_scalars('Valid PSNR', {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr}, step)
            writer.add_scalars('Valid SSIM', {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim}, step)
            
            end = time()
            elpased_time = end - start
            print(f'Epoch: {epoch+1}/{epochs} | Time: {elpased_time:.3f} | Val Loss: {avg_loss:.3f} | Val PSNR: {avg_psnr:.3f} | Val SSIM: {avg_ssim:.3f}')

            total_time += elpased_time

            diff = avg_psnr - avg_bicubic_psnr
            if diff > best_psnr_diff:
                best_psnr_diff = diff
                torch.save(net.state_dict(), check_pnt_path)
        
        epoch += 1
            
        states = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr_diff': best_psnr_diff,
            'epoch': epoch,
            'total_time': total_time
        }

        torch.save(states, last_pnt_path)
