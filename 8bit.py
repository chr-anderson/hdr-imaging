import cv2 as cv
import numpy as np

# -------- Load exposure images into a list --------
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)  # 32-bit float data type

# -------- Merge exposures into an HDR image --------
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# -------- Map the 32-bit data into the range [0,1]
tonemap1 = cv.createTonemap(gamma=2.2)  # Applying gamma correction to account for human light perception
res_robertson = tonemap1.process(hdr_robertson.copy())

# -------- Convert to 16-bit and save --------
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
cv.imwrite("hdr_robertson_8bit.jpg", res_robertson_8bit)

#cv.imshow("Results", res_robertson)
