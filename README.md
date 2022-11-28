# SwinIR 
<p> Reimplementation of 4x SwinIR https://arxiv.org/abs/2108.10257 </p>

## 4x Result

### PSNR
|Dataset|Bicubic|SwinIR|
|:---:|:---:|:---:|
|Set5|28.648|32.666 (32.920)|
|Set14|26.406|29.418 (29.090)|
|Urban100|23.220|27.133 (27.450)|
<p>(number) comes from the paper.</p>

### Set14
| Bicubic | SwinIR | GT |
|:---:|:---:|:---:|
|<img width="264" alt="image" src="https://user-images.githubusercontent.com/53179332/204194615-cb470e05-fd2a-46cc-aae3-0ac42f0ec7b5.png">|<img width="264" alt="image" src="https://user-images.githubusercontent.com/53179332/204194625-29485e22-73e4-4def-a8ef-eb393f06c8c8.png">|<img width="265" alt="image" src="https://user-images.githubusercontent.com/53179332/204194543-d00bd079-1348-4f96-b929-c7a7a52fd042.png">|



## Train Setting
|Item|Setting|
|:---:|:---:|
|Train Data|DIV2K, Flickr2K|
|Preprocess|[-1,1] Normalization |
|Random Transforms|Crop {DIV2K(48x48), Flickr2K(64x64)}, Rotation {90 degree} |
|Validation Data|DIV2K|
|Test Data| Set5, Set14, Urban100|
|Scale| 4x |
|Optimizer|Adam|
|Learning Rate|2e-4|
|Scheduler|Reduce LR to half at 50%, 80%, 90%, 95% of 5e5 iterations|
|Actual Trained Iterations|Around 4.3e5|
|Loss|L1|
|Batch|8 {2 for each GPU, total 4 GPUs are utilized} |
