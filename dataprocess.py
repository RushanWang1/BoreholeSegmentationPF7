import rasterio 
import numpy as np
# from rasterio.plot import show
from rasterio.windows import Window
import matplotlib.image as image
import matplotlib.pyplot as plt
from PIL import Image

############ THIS PART OF THE CODE IS FOR MERGING DIFFERENT LABEL FILE INTO SINGLE LAYER OF TIFF FILE ############ 
# read breakout and faultzone label
with rasterio.open("data/refined/210202_fullyfailedbreakout.tif") as img1:
    # print(img1.shape)
    # window = Window(0,0,4000,4000)
    data_breakout = img1.read()
    # data_breakout[data_breakout == 0] = 99
    # data_breakout[data_breakout == 1] = 4
    # data_breakout[data_breakout == 2] = 5
    # data_breakout[data_breakout == 1] = 4
    # data_breakout[data_breakout == 2] = 5
    # values1 = np.unique(data_breakout)
    print(np.unique(data_breakout))
    # show(data)

with rasterio.open("data/refined/alllabel_210202_new.tif") as img3:
    # print(img1.shape)
    # window = Window(0,0,4000,4000)
    profile = img3.profile
    allLabel = img3.read()
    # data_cracks[data_cracks == 0] = 99
    # data_breakout[data_breakout == 1] = 4
    # data_breakout[data_breakout == 2] = 5
    allLabel[allLabel == 4] = 0
    values1 = np.unique(allLabel)
    print(values1)

# read cracks label  
# with rasterio.open("data/temporal_compare_data/201113_label/tectonic_desiccate.tif") as img2:
#     # print(img2.shape)
#     # window = Window(0,0,4000,4000)
#     profile = img2.profile
#     data_cracks = img2.read()
#     # data_cracks[data_cracks == 3] = 99
#     values2 = np.unique(data_cracks)
#     print(values2)


# merge label to get the overall groundtruth
# allfaultzone = np.maximum(data_breakout,data_cracks)
# allLabel_crack = np.maximum(data_cracks, data_faultgouge)
# allLabel[data_cracks == 1] = 1
# allLabel[data_cracks == 2] = 2
allLabel = np.maximum(data_breakout, allLabel)
# allLabel[allLabel==99]=0
print(allLabel[0].shape)
print(np.unique(allLabel))
# show(allLabel, cmap='Pastel1')
with rasterio.open('data/refined/alllabel_210202_fullybreak.tif', 'w', **profile) as dst:
    dst.write(allLabel[0].astype(rasterio.uint8), 1)

print("Finish writing new label!")


############ THIS PART OF THE CODE IS FOR CROPPING THE LARGE IMAGE OR LABEL INTO SMALLER PATCH SIZE ############ 

# Cropping both image and label into 1024x1024 size, creating dataset
# height = 512
# width = 512
# col = 0
# row = 0
# n=0
# x = 0
# y = 0
# # Over lap croping, 512*512 image size, interval 256
# with rasterio.open("data/refined/alllabel_210202_new.tif") as img:
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

####### Build temporal compare dataset
# width = 512
# height = 512
# n = 0
# date = 220621
# with rasterio.open(f"data/temporal_compare_data/new_image/{date}_cropped_image.tif") as img:
#     # print(img.shape)  
#     col_num = img.width//(width - 20)+1
#     row_num = img.height//(height - 20)+1
#     for y in range(row_num):
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
#             image.imsave(f'data/temporal_compare_data/new_image/{date}_image_{width}_overlap20/image_{n}.jpg', np.stack((data_gt[0],data_gt[1],data_gt[2]),axis = 2))
#             n+=1

###### build training and testing dataset of different crop size ######
# width = 512
# height = 512
# n = 0
# with rasterio.open("data/temporal_compare_data/new_image/210720_cropped_image.tif") as img:
#     print(img.shape)  
#     image_width = img.width
#     image_height = img.height
#     image_offset = height/2
#     # col_num = img.width//(width - 20)+1
#     # row_num = img.height//(height - 20)+1
#     row_num = int(image_height/image_offset)
#     col_num = int(image_width/image_offset)
#     for y in range(row_num):
#         for x in range (col_num):
#             if x <(col_num-1):
#                 col = x*image_offset               
#             elif x==(col_num-1):
#                 col = img.width - width                 
#             if y<(row_num-1):
#                 row = y*image_offset
#             elif y==(row_num-1):
#                 row = img.height - height
#             currentwindow = Window(col,row,width,height)
#             data_gt = img.read(window = currentwindow)
#             print(int(col),int(row))
#             # radius = rescale_radius[0,int(row):int(row)+512, int(col):int(col)+512]
#             # image.imsave(f'FineTune/data/image/image_{n}.png', data_gt[0])
#             image.imsave(f'data/temporal_compare_data/new_image/210720_image_{width}/image_{n}.jpg', np.stack((data_gt[0],data_gt[1],data_gt[2]),axis = 2))
#             n+=1
# n = 0

# with rasterio.open("data/refined/test_alllabel_210202_new.tif") as img:
#     # print(img.shape)
#     image_width = img.width
#     image_height = img.height  
#     # window_test = Window(0,0,image_width, int(image_height*0.2))
#     # window_train = Window(0,int(image_height*0.2),image_width, image_height - int(image_height*0.2))
#     # img = img.read(window  = window_train)
    
#     row_num = int(image_height/(height/2))
#     col_num = int(image_width/(width/2))
#     for y in range(row_num):
#         for x in range (col_num):
#             if x <(col_num-1):
#                 col = x*(width/2)                
#             elif x==(col_num-1):
#                 col = img.width - width                 
#             if y<(row_num-1):
#                 row = y*(height/2)
#             elif y==(row_num-1):
#                 row = img.height - height
#             window = Window(col,row,width,height)
#             data_gt = img.read(window = window)
#             array = Image.fromarray(data_gt[0].astype(np.uint8),'L')
#             array.save(f'data/test_annotation_new_{width}/annotation_{n}.png')
#             # np.savetxt(f'FineTune/data/annotation/annotation_{n}.csv', data_gt[0],delimiter=",")
#             # image.imsave(f'FineTune/data/annotation/annotation_{n}.tif', data_gt[0])
#             n+=1 


# Create one hot encoding from the label with extra class in fault 
# Define the one-hot encoding for each class
# one_hot_map = {
#     0: [1, 0, 0, 0, 0, 0],
#     1: [0, 1, 0, 0, 0, 0],
#     2: [0, 0, 1, 0, 0, 0],
#     3: [0, 0, 0, 1, 0, 0],
#     4: [0, 0, 0, 0, 1, 0],
#     5: [0, 0, 0, 0, 0, 1],
#     6: [0, 0, 0, 0, 1, 1],
#     7: [0, 0, 1, 0, 0, 1],
# }

# # Load the TIFF file
# tiff_path = 'data/refined/alllabel_faultzone.tif'
# image = Image.open(tiff_path)
# label_array = np.array(image)

# # Get the dimensions of the image
# height, width = label_array.shape

# # Initialize the one-hot encoded array
# one_hot_encoded = np.zeros((height, width, 6), dtype=np.uint8)

# # Apply the one-hot encoding
# for i in range(height):
#     for j in range(width):
#         one_hot_encoded[i, j] = one_hot_map[label_array[i, j]]

# # If you need to save the result as an image, you can save each channel as a separate image or combine them as needed
# # Example: Saving one of the channels
# # output_image = Image.fromarray(one_hot_encoded[:, :, 0])
# # output_image.save('one_hot_encoded_channel_0.tif')

# # Alternatively, you can save the entire array using np.save to preserve all channels
# np.save('one_hot_encoded.npy', one_hot_encoded)

# print("Done!")
