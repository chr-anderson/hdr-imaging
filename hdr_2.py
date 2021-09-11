import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -------- Load exposure images into a list --------
files = np.loadtxt("phone_list.txt", dtype=str)  # Load a .txt file containing names in one column and exposures in another

img_fn = [row[0] for row in files]
img_list = [cv.imread(fn) for fn in img_fn]

exposure_times = [row[1] for row in files]
exposure_times = np.array(exposure_times, dtype=np.float32)  # 32-bit float data type

# -------- Merge exposures into an HDR image --------
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# -------- Estimate the camera response function --------
cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)
blues = [row[0][0] for row in crf_robertson]
greens = [row[0][1] for row in crf_robertson]
reds = [row[0][2] for row in crf_robertson]
plt.plot(blues, c='blue')
plt.plot(greens, c='green')
plt.plot(reds, c='red')
plt.show()

# -------- Map the 32-bit data into the range [0,1]
# TODO: Try different tone mapping algorithms. Most of the noise appears to be introduced here
tonemap1 = cv.createTonemap(gamma=2.5)  # Applying gamma correction to account for human light perception
res_robertson = tonemap1.process(hdr_robertson.copy())

# -------- Convert to 16-bit and save --------
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
cv.imwrite("hdr_phone.jpg", res_robertson_8bit)

cv.imshow("Results", res_robertson_8bit)
cv.waitKey(0)
