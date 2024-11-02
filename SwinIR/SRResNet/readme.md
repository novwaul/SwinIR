# SRResNet
The 4x generator of SRGAN https://github.com/novwaul/SRGAN

<img width="1104" alt="스크린샷 2022-10-08 오후 12 03 17" src="https://user-images.githubusercontent.com/53179332/194684972-dda9227e-e99c-40dd-ac47-445afb31e8b9.png">

## Train Setting
|Item|Setting|
|:---:|:---:|
|Train Data|DIV2K|
|Validation Data|DIV2K|
|Test Data| Set5, Set14, Urban100|
|Scale| 4x |
|Loss|L1|
|Optimizer|Adam|
|Scheduler|Multistep(g=0.5, m=[0.75 x epochs, 0.90 x epochs])|
|Iterations|2e5|
|Batch|8|
