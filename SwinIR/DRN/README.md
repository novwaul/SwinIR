# Dual Regression Net
<p> Reimplementation of 4x DRN https://arxiv.org/abs/2003.07018 </p>

## 4x Result

### PSNR
|Dataset|Bicubic|DRN|
|:---:|:---:|:---:|
|Set5|28.648|32.503 (-0.237)|
|Set14|26.406|28.830 (-0.150)|
|Urban100|23.220|26.677 (-0.353)|
<p>(number) means PSNR difference compared to the paper. </p>
<p>[2023.05.04: Set14 result update]</p>
<p>Now gray-scale test image result is included. Please be aware that still this code does not use gray-scale test images.</p>


### Set14
| GT | Bicubic | DRN |
|:---:|:---:|:---:|
|<img width="159" alt="image" src="https://user-images.githubusercontent.com/53179332/198077414-7ac03b47-56ee-4af5-bd83-508841c2551c.png">|<img width="159" alt="image" src="https://user-images.githubusercontent.com/53179332/198077493-ad9017c7-46c5-4f68-afb1-c5e3736890a8.png">|<img width="160" alt="image" src="https://user-images.githubusercontent.com/53179332/198077589-4ce57b59-7c1c-43b3-95c4-61716cb67fad.png">|



## Train Setting
|Item|Setting|
|:---:|:---:|
|Train Data|DIV2K, PASCAL VOC|
|Crop|32 x 32|
|Validation Data|DIV2K|
|Test Data| Set5, Set14, Urban100|
|Scale| 4x |
|Optimizer|Adam|
|Scheduler|1e-4 to 1e-7 with Cosine Annealing|
|Iterations|Around 3e5|
|Batch|16|
