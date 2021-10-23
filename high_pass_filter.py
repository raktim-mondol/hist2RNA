import cv2
import numpy

img = cv2.imread('./TCGA-A2-A0T2-DX1_xmin63557_ymin56751_MPP-0.2500.png')



def high_pass_filter(image, kernel_size=50):
    size =kernel_size
    if not size%2:
        size +=1
    kernel = numpy.ones((size,size),numpy.float32)/(size*size)
    filtered= cv2.filter2D(img,-1,kernel)
    filtered = img.astype('float32') - filtered.astype('float32')
    output = (filtered + 127*numpy.ones(img.shape, numpy.uint8)).astype('uint8')
    cv2.imwrite('./TCGA-A2-A0T2-DX1_xmin63557_ymin56751_MPP-0.2500_HF.png', output)



high_pass_filter(img)







