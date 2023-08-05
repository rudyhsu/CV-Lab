import cv2
import numpy as np
import json

pixel_coords_src = []
pixel_coords_dst = []
pixel_coords_num = 1 

# Get pixel coordinate
def click_event_src(event, x, y, flags, params):
    global pixel_coords_src, pixel_coords_num

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})' + 'Source Coordinate number : ' + str(pixel_coords_num))
        pixel_coords_num += 1
        pixel_coords_src.append([int((float)(x)), int((float)(y))])

def click_event_dst(event, x, y, flags, params):
    global pixel_coords_dst, pixel_coords_num

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})' + 'Dst Coordinate number : ' + str(pixel_coords_num))
        pixel_coords_num += 1
        pixel_coords_dst.append([int((float)(x)), int((float)(y))])



# Read source image.
im_src = cv2.imread('./input_right_top/img/scene07851.png')
print(im_src.shape)

im_dst = cv2.imread('./input_right_top/img/sat.png')
print(im_dst.shape)
# create a window for src
cv2.namedWindow('Src Point Coordinates')
# bind the callback function to window
cv2.setMouseCallback('Src Point Coordinates', click_event_src)
# display the image
pixel_coords_num = 1
cv2.imshow('Src Point Coordinates', im_src)
cv2.waitKey(0)
pts_src = np.array(pixel_coords_src)

# show the selected points on src image
sel_img = np.copy(im_src)
for i in range(pts_src.shape[0]):
    x = pts_src[i][0]
    y = pts_src[i][1]
    cv2.circle(sel_img, (x,y), 1, (0,255,255), -1)
    cv2.putText(sel_img, f'pts:{i}',(x,y),
    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
cv2.imshow('Selected Point Coordinates', sel_img)

# create a window for dst
cv2.namedWindow('Dst Point Coordinates')
# bind the callback function to window
cv2.setMouseCallback('Dst Point Coordinates', click_event_dst)
# display the image
pixel_coords_num = 1
cv2.imshow('Dst Point Coordinates', im_dst)
cv2.waitKey(0)
pts_dst = np.array(pixel_coords_dst)


# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# save h
with open('./input_right_top/matrix/mat.json') as f:
    data = {}
data['mat'] = []
data['mat'].append(h.tolist())
with open('./input_right_top/matrix/mat.json', 'w') as f:
    json.dump(data, f)
# # Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

# # Display images
# cv2.imshow("Source Image", im_src)
# cv2.imshow("Destination Image", im_dst)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)

