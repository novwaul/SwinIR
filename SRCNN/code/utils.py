import numpy as np

def getPSNR(img_np, lbl_np):
    diff = img_np - lbl_np
    mse = np.mean(diff**2)
    return -10*np.log10(mse)

##fix?
def getSSIM(img_np, lbl_np):
    ux = np.mean(img_np)
    sx = np.std(img_np, ddof=1)
    uy = np.mean(lbl_np)
    sy = np.std(lbl_np, ddof=1)
    sxy = np.sum( (img_np-ux) * (lbl_np-uy) ) / (img_np.size - 1)
    c1 = (0.01)**2
    c2 = (0.03)**2
    return ((2*ux*uy + c1)*(2*sxy + c2)) / ((ux**2 + uy**2 + c1)*(sx**2 + sy**2 + c2))

def RGB2Y(img_np):
   return (16.0 + 65.481*img_np[:,0,:,:] + 128.553*img_np[:,1,:,:] + 24.966*img_np[:,2,:,:]) / 255.0
