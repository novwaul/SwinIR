# SRGAN

Reimpelmentation of SRGAN https://arxiv.org/abs/1609.04802

This code use custom SRResNet https://github.com/novwaul/SRResNet

## 4x Result

### PSNR
|Dataset|Bicubic|SRResNet|SRGAN|
|:---:|:---:|:---:|:---:|
|Set5|28.648|30.961 (-1.098)|28.683 (-0.717)|
|Set14|26.406|27.870 (-0.620)|26.020 (Same)|
|Urban100|23.220|24.759 (None)|23.300 (None)|
<p>(number) means PSNR difference compared to the paper.</p>
<p>[2023.05.04: Set14 result update]</p>
<p>Now gray-scale test image result is included. Please be aware that still this code does not use gray-scale test images.</p>

### Set14
| GT | Bicubic | SRResNet | SRGAN |
|:---:|:---:|:---:|:---:|
|<img width="159" alt="image" src="https://user-images.githubusercontent.com/53179332/198077414-7ac03b47-56ee-4af5-bd83-508841c2551c.png">|<img width="159" alt="image" src="https://user-images.githubusercontent.com/53179332/198077493-ad9017c7-46c5-4f68-afb1-c5e3736890a8.png">|![image](https://user-images.githubusercontent.com/53179332/198231670-27b0da27-c4f2-43ad-9252-42957e517de9.png)|![image](https://user-images.githubusercontent.com/53179332/198231710-2ce9ab90-6aa8-4839-806e-cab4a4f99597.png)|



## Train Setting
<p>This code use pretrained SRResNet. For pretraining, note https://github.com/novwaul/SRResNet</p>

|Item|Setting|
|:---:|:---:|
|Train Data|DIV2K, PASCAL VOC|
|Crop|24 x 24|
|Validation Data|DIV2K|
|Test Data| Set5, Set14, Urban100|
|Scale| 4x |
|Optimizer|Adam|
|Learning Rate|1e-4 for [0, 1e5], 1e-5 for [1e5, 3e5] iterations|
|Iterations|Around 3e5|
|Batch|16|
