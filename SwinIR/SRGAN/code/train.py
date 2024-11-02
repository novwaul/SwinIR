import torch
from time import time
from utils import calc_psnr, calc_ssim, cvrt_rgb_to_y, norm, denorm
    
def train(args, resume):

    generator = args['generator']
    disciminator = args['discriminator']
    bicubic = args['bicubic']

    d_optimizer = args['d_optimizer']
    g_optimizer = args['g_optimizer']
    d_scheduler = args['d_scheduler']
    g_scheduler = args['g_scheduler']

    adversarial_loss = args['adversarial_loss']
    content_loss = args['content_loss']
    pixel_loss = args['pixel_loss']

    device = args['device']
    epochs = args['epochs']
    crop_out = args['crop_out']

    train_dataloaders = args['train_dataloaders']
    valid_dataloader = args['valid_dataloader']

    check_pnt_path = args['check_pnt_path']
    last_pnt_path = args['last_pnt_path']
    old_pnt_path = args['old_pnt_path']
    resnet_path = args['resnet_path']

    writer = args['writer']

    if resume:
        states = torch.load(last_pnt_path)

        generator.load_state_dict(states['generator'])
        g_optimizer.load_state_dict(states['g_optimizer'])
        g_scheduler.load_state_dict(states['g_scheduler'])

        min_g_loss = states['min_g_loss']

        disciminator.load_state_dict(states['discriminator'])
        d_optimizer.load_state_dict(states['d_optimizer'])
        d_scheduler.load_state_dict(states['d_scheduler'])

        epoch = states['epoch']
        total_time = states['total_time']
    else:
        generator.load_state_dict(torch.load(resnet_path))
        min_g_loss = 100
        epoch = 0
        total_time = 0.0


    total_iterations = sum([len(train_dataloader) for train_dataloader in train_dataloaders])
    batch_num = len(valid_dataloader)

    while epoch < epochs:

        start = time()

        step = epoch*total_iterations
        
        disciminator.train()
        generator.train()

        for i, train_dataloader in enumerate(train_dataloaders):
            for j, (img, lbl) in enumerate(train_dataloader):
                iteration = sum([len(train_dataloaders[i]) for i in range(0, i)]) + j
            
                img = norm(img.to(device))
                lbl = norm(lbl.to(device))
                
                batch_size, *_ = lbl.shape
                shape = (batch_size, 1)
                real_lbl = torch.ones(shape, dtype=lbl.dtype, device=device)
                fake_lbl = torch.zeros(shape, dtype=lbl.dtype, device=device)

                '''
                #Discriminator Train
                '''
                d_optimizer.zero_grad()

                out = generator(img)

                real = lbl
                fake = out.detach()

                real_out = disciminator(real)
                fake_out = disciminator(fake)

                d_r_loss = adversarial_loss(real_out, real_lbl)
                d_f_loss = adversarial_loss(fake_out, fake_lbl)
                d_loss = d_r_loss + d_f_loss
                d_loss.backward()

                d_optimizer.step()
                d_scheduler.step() 
                
                '''
                #Generator Train
                '''

                g_optimizer.zero_grad()
                
                out = generator(img)
                fake_out = disciminator(out)

                g_adv_loss = 1e-3*adversarial_loss(fake_out, real_lbl)
                g_cnt_loss = 0.006*content_loss(out, lbl)
                g_pxl_loss = pixel_loss(out,lbl)

                g_loss = g_adv_loss + g_cnt_loss + g_pxl_loss
                g_loss.backward()
                
                g_optimizer.step()
                g_scheduler.step()
        

                '''
                #Summary
                '''
                    
                if iteration%400 == 399:
                    writer.add_scalars('Train Discriminator Loss', {'Total': d_loss.item(), 'Real': d_r_loss.item(), 'Fake': d_f_loss.item()}, step+iteration)

                    out_cpu = denorm(out.detach()).clamp(min=0.0, max=1.0).to('cpu')
                    img_cpu = bicubic(denorm(img.detach())).clamp(min=0.0, max=1.0).to('cpu')
                    lbl_cpu = denorm(lbl.detach()).to('cpu')

                    out_y_np = cvrt_rgb_to_y(out_cpu.numpy())
                    img_y_np = cvrt_rgb_to_y(img_cpu.numpy())
                    lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                    psnr = calc_psnr(out_y_np, lbl_y_np, crop_out)
                    ssim = calc_ssim(out_y_np, lbl_y_np, crop_out)
                    bicubic_psnr = calc_psnr(img_y_np, lbl_y_np, crop_out)
                    bicubic_ssim = calc_ssim(img_y_np, lbl_y_np, crop_out)

                    writer.add_scalars('Train Generator Loss', {'Total': g_loss.item(), 'MSE': g_pxl_loss.item(), 'GAN': g_adv_loss.item(), 'VGG': g_cnt_loss.item()},  step+iteration)
                    writer.add_scalars('Train PSNR', {'Model PSNR': psnr, 'Bicubic PSNR': bicubic_psnr}, step+iteration)
                    writer.add_scalars('Train SSIM', {'Model SSIM': ssim, 'Bicubic SSIM': bicubic_ssim}, step+iteration)
                    
                    print(f'Epoch: {epoch+1}/{epochs} | {iteration+1}/{total_iterations} | D Loss: {d_loss.item():.3f} | G Loss: {g_loss.item():.3f} | PSNR: {psnr:.3f} | SSIM: {ssim:.3f}')

        disciminator.eval()
        generator.eval()

        with torch.no_grad():

            total_d_loss = 0.0
            total_d_r_loss = 0.0
            total_d_f_loss = 0.0

            total_g_loss = 0.0
            total_g_adv_loss = 0.0
            total_g_cnt_loss = 0.0
            total_g_pxl_loss = 0.0

            total_psnr = 0.0
            total_ssim = 0.0
            total_bicubic_psnr = 0.0
            total_bicubic_ssim = 0.0

            for iteration, (img, lbl) in enumerate(valid_dataloader):

                batch_size, *_ = lbl.shape
                shape = (batch_size, 1)

                real_lbl = torch.ones(shape, dtype=lbl.dtype, device=device)
                fake_lbl = torch.zeros(shape, dtype=lbl.dtype, device=device)

                img = norm(img.to(device))
                lbl = norm(lbl.to(device))
                out = generator(img)
                real_out = disciminator(lbl)
                fake_out = disciminator(out)

                d_r_loss = adversarial_loss(real_out, real_lbl)
                d_f_loss = adversarial_loss(fake_out, fake_lbl)

                d_loss = d_r_loss + d_f_loss

                total_d_loss += d_loss.item()
                total_d_r_loss += d_r_loss.item()
                total_d_f_loss += d_f_loss.item()


                g_adv_loss = 1e-3*adversarial_loss(fake_out, real_lbl)
                g_cnt_loss = 0.006*content_loss(out, lbl)
                g_pxl_loss = pixel_loss(out, lbl)

                g_loss = g_adv_loss + g_cnt_loss + g_pxl_loss

                out_cpu = denorm(out).clamp(min=0.0, max=1.0).to('cpu')
                img_cpu = bicubic(denorm(img)).clamp(min=0.0, max=1.0).to('cpu')
                lbl_cpu = denorm(lbl).to('cpu')
                lr_cpu = denorm(img).to('cpu')
                if iteration == 0 and (epoch%5 == 4 or epoch == 0):
                    writer.add_images(tag='Valid Upscale/A. Input', img_tensor=lr_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/B. Ground Truth', img_tensor=lbl_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/C. Bicubic', img_tensor=img_cpu, global_step=epoch+1)
                    writer.add_images(tag='Valid Upscale/D. Model', img_tensor=out_cpu, global_step=epoch+1)

                out_y_np = cvrt_rgb_to_y(out_cpu.numpy())
                img_y_np = cvrt_rgb_to_y(img_cpu.numpy())
                lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

                psnr = calc_psnr(out_y_np, lbl_y_np, crop_out)
                ssim = calc_ssim(out_y_np, lbl_y_np, crop_out)
                bicubic_psnr = calc_psnr(img_y_np, lbl_y_np, crop_out)
                bicubic_ssim = calc_ssim(img_y_np, lbl_y_np, crop_out)

                total_g_loss += g_loss.item()
                total_g_adv_loss += g_adv_loss.item()
                total_g_cnt_loss += g_cnt_loss.item()
                total_g_pxl_loss += g_pxl_loss.item()

                total_psnr += psnr
                total_ssim += ssim
                total_bicubic_psnr += bicubic_psnr
                total_bicubic_ssim += bicubic_ssim

            '''
            #Summary
            '''

            avg_d_loss = total_d_loss/batch_num
            avg_d_r_loss = total_d_r_loss/batch_num
            avg_d_f_loss = total_d_f_loss/batch_num
            writer.add_scalars('Valid Discriminator Loss', {'Total': avg_d_loss, 'Real': avg_d_r_loss, 'Fake': avg_d_f_loss}, step)

            avg_g_loss = total_g_loss/batch_num
            avg_g_adv_loss = total_g_adv_loss/batch_num
            avg_g_cnt_loss = total_g_cnt_loss/batch_num
            avg_g_pxl_loss = total_g_pxl_loss/batch_num

            avg_psnr = total_psnr/batch_num
            avg_ssim = total_ssim/batch_num
            avg_bicubic_psnr = total_bicubic_psnr/batch_num
            avg_bicubic_ssim = total_bicubic_ssim/batch_num
        
            writer.add_scalars('Valid Generator Loss', {'Total': avg_g_loss, 'MSE': avg_g_pxl_loss, 'GAN': avg_g_adv_loss, 'VGG': avg_g_cnt_loss}, step)
            writer.add_scalars('Valid PSNR', {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr}, step)
            writer.add_scalars('Valid SSIM', {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim}, step)
            
            end = time()
            elpased_time = end - start

            print(f'Epoch: {epoch+1}/{epochs} | Time: {elpased_time:.3f} | Val D Loss: {avg_d_loss:.3f} | Val G Loss: {avg_g_loss:.3f} | Val PSNR: {avg_psnr:.3f} | Val SSIM: {avg_ssim:.3f}')

            if avg_g_loss < min_g_loss:
                min_g_loss = avg_g_loss
                torch.save(generator.state_dict(), check_pnt_path)

            total_time += elpased_time

        if epoch > 0:
            old_states = torch.load(last_pnt_path)
            torch.save(old_states, old_pnt_path)
        
        epoch += 1

        states = {
            'discriminator': disciminator.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'd_scheduler': d_scheduler.state_dict(),
            'generator': generator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'g_scheduler': g_scheduler.state_dict(),
            'min_g_loss': min_g_loss,
            'epoch': epoch,
            'total_time': total_time
        }

        torch.save(states, last_pnt_path)
