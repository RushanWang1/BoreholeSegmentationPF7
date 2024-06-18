import rasterio 
import numpy as np
# from rasterio.plot import show
from rasterio.windows import Window
import matplotlib.image as image
import matplotlib.pyplot as plt
from PIL import Image

# read breakout and faultzone label
# with rasterio.open("data/refined/breakout.tif") as img1:
#     # print(img1.shape)
#     # window = Window(0,0,4000,4000)
#     data_breakout = img1.read()
#     data_breakout[data_breakout == 0] = 99
#     data_breakout[data_breakout == 1] = 4
#     data_breakout[data_breakout == 2] = 5
#     data_breakout[data_breakout == 4] = 6
#     values1 = np.unique(data_breakout)
#     print(values1)
#     # show(data)

# # read cracks label  
# with rasterio.open("data/refined/cracks.tif") as img2:
#     # print(img2.shape)
#     # window = Window(0,0,4000,4000)
#     profile = img2.profile
#     data_cracks = img2.read()
#     data_cracks[data_cracks == 0] = 99
#     values2 = np.unique(data_cracks)
#     print(values2)


# # merge label to get the overall groundtruth
# allLabel = np.minimum(data_breakout,data_cracks)
# allLabel[allLabel==99]=0
# print(allLabel[0].shape)
# print(np.unique(allLabel))
# # show(allLabel, cmap='Pastel1')
# with rasterio.open('data/refined/alllabel_incipient.tif', 'w', **profile) as dst:
#     dst.write(allLabel[0].astype(rasterio.uint8), 1)

# Cropping both image and label into 1024x1024 size, creating dataset
height = 512
width = 512
col = 0
row = 0
n=0
x = 0
y = 0
# Over lap croping, 512*512 image size, interval 256
# with rasterio.open("data/refined/alllabel_incipient.tif") as img:
#     # print(img.shape)  
#     for y in range(27):
#         for x in range (189):
#             if x <188:
#                 col = x*(width/2)                
#             elif x==188:
#                 col = img.width - width                 
#             if y<26:
#                 row = y*(height/2)
#             elif y==26:
#                 row = img.height - height
#             window = Window(col,row,width,height)
#             data_gt = img.read(window = window)
#             array = Image.fromarray(data_gt[0].astype(np.uint8),'L')
#             array.save(f'data/annotation_overlap_incipient/annotation_{n}.png')
#             # np.savetxt(f'FineTune/data/annotation/annotation_{n}.csv', data_gt[0],delimiter=",")
#             # image.imsave(f'FineTune/data/annotation/annotation_{n}.tif', data_gt[0])
#             n+=1

# Over lap croping, 512*512 image size, interval 256
# with rasterio.open("data/refined/RadiusCropped.tif") as radiusimg:
#     radius_array = radiusimg.read()
#     print(radius_array.shape)
# normalizedData = (radius_array-np.min(radius_array))/(np.max(radius_array)-np.min(radius_array))
# rescale_radius = normalizedData*255
# rescale_radius = rescale_radius.astype(int)
# width = 1024
# height = 1024
# with rasterio.open("data/temporal_compare_data/230816_image.tif") as img:
#     # print(img.shape)  
#     col_num = img.width//(width - 20)
#     row_num = img.height//(height - 20)
#     for y in range(15):
#         for x in range (col_num):
#             if x < (col_num-1):
#                 col = x*(width - 20)               
#             elif x==(col_num-1):
#                 col = img.width - width                 
#             if y<(row_num-1):
#                 row = y*(height - 20)
#             elif y==(row_num-1):
#                 row = img.height - height
#             currentwindow = Window(col,row,width,height)
#             data_gt = img.read(window = currentwindow)
#             print(int(col),int(row))
#             # radius = rescale_radius[0,int(row):int(row)+512, int(col):int(col)+512]
#             # image.imsave(f'FineTune/data/image/image_{n}.png', data_gt[0])
#             image.imsave(f'data/temporal_compare_data/230816_image_1024_overlap_20/image_{n}.jpg', np.stack((data_gt[0],data_gt[1],data_gt[2]),axis = 2))
#             n+=1
width = 512
height = 512
with rasterio.open("data/refined/test_croppedimage.tif") as img:
    # print(img.shape)  
    image_width = img.width
    image_height = img.height
    row_num = int(image_height/(height/2))
    col_num = int(image_width/(width/2))
    for y in range(row_num):
        for x in range (col_num):
            if x <(col_num-1):
                col = x*(width/2)                
            elif x==(col_num-1):
                col = img.width - width                 
            if y<(row_num-1):
                row = y*(height/2)
            elif y==(row_num-1):
                row = img.height - height
            currentwindow = Window(col,row,width,height)
            data_gt = img.read(window = currentwindow)
            print(int(col),int(row))
            # radius = rescale_radius[0,int(row):int(row)+512, int(col):int(col)+512]
            # image.imsave(f'FineTune/data/image/image_{n}.png', data_gt[0])
            image.imsave(f'data/test_image_512/image_{n}.jpg', np.stack((data_gt[0],data_gt[1],data_gt[2]),axis = 2))
            n+=1
n = 0
with rasterio.open("data/refined/test_alllabel.tif") as img:
    # print(img.shape)  
    image_width = img.width
    image_height = img.height
    row_num = int(image_height/(height/2))
    col_num = int(image_width/(width/2))
    for y in range(row_num):
        for x in range (col_num):
            if x <(col_num-1):
                col = x*(width/2)                
            elif x==(col_num-1):
                col = img.width - width                 
            if y<(row_num-1):
                row = y*(height/2)
            elif y==(row_num-1):
                row = img.height - height
            window = Window(col,row,width,height)
            data_gt = img.read(window = window)
            array = Image.fromarray(data_gt[0].astype(np.uint8),'L')
            array.save(f'data/test_annotation_512/annotation_{n}.png')
            # np.savetxt(f'FineTune/data/annotation/annotation_{n}.csv', data_gt[0],delimiter=",")
            # image.imsave(f'FineTune/data/annotation/annotation_{n}.tif', data_gt[0])
            n+=1 
# import matplotlib.pyplot as plt
# im = plt.imread('FineTune/data/annotation/annotation_1.png')
# print(im.shape)

print("Done!")
