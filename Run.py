
from lib import ClassAverages
from yolo.yolo import cv_Yolo

import json
import os
import time

import numpy as np
import cv2

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/img/",
                    help="The dir path of the testing dataset")

parser.add_argument("--matrix-path", default="homo_matrix/mat.json",
                    help="The file path of the homographic matrix input")

parser.add_argument("--check-projection", action="store_true",
                    help="Project the predicted position onto BEV image")

parser.add_argument("--check-detection", action="store_true",
                    help="Plot 2D bounding box on input images")

# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    cv2.line(img, pt2, pt3, (0, 0, 255), 2)
    cv2.line(img, pt3, pt4, (0, 0, 255), 2)
    cv2.line(img, pt4, pt1, (0, 0, 255), 2)

def main():
    FLAGS = parser.parse_args()

    # create json for storing detected center
    open('center.json', 'w').close()
    with open('center.json') as f:
        try : 
            data = json.load(f)
        except : 
            data = {}
    data["center"] = []
    with open('center.json', 'w') as f:
        json.dump(data, f)

    # load in the homographic matrix
    mat_path = FLAGS.matrix_path
    with open(mat_path) as f:
        data = json.load(f)
    homo_matrix = np.array(data['mat']).reshape((3,3))

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    image_dir = FLAGS.image_dir
    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        if FLAGS.check_projection :
            # check_img = np.copy(truth_img)
            # im_dst = cv2.imread('./checking/sat.png')
            # check_img = cv2.warpPerspective(check_img, homo_matrix, (im_dst.shape[1], im_dst.shape[0]))
            check_path = os.path.abspath(os.path.dirname(__file__)) + '/checking/sat/sat.png'
            check_img = cv2.imread(check_path)

        detections = yolo.detect(yolo_img)

        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue

            box_2d = detection.box_2d
            detected_class = detection.detected_class

            if FLAGS.check_projection:
                plot_2d_box(img, box_2d)
            
            pt1, pt2, pt3, pt4 = create_2d_box(box_2d)
            center = []
            center.append(int((pt1[0] + pt2[0] + pt3[0] + pt4[0])/4))
            center.append(int((pt1[1] + pt2[1] + pt3[1] + pt4[1])/4))

            # calculate the coordinate of the projected center
            center.append(1)
            center = np.array(center).reshape((3,1))
            transformed_center = np.matmul(homo_matrix, center)
            transformed_center = np.divide(transformed_center[:2], transformed_center[2]).astype(int)

            # check if the prediction is accurate by plotting it
            if FLAGS.check_projection:
                x = transformed_center[0][0]
                y = transformed_center[1][0]
                cv2.circle(check_img, (x,y), 3, (0,255,255), -1)

            # dict of the current detection
            dict_center = {}
            dict_center['pt'] = [float(transformed_center[0][0]), float(transformed_center[1][0])]
            dict_center['class'] = detected_class
            dict_center['fid'] = str(int(img_id))

            with open('center.json') as f:
                data = json.load(f)

            data['center'].append(dict_center)

            with open('center.json', 'w') as f:
                json.dump(data, f)
        print("\n")
        print('Got %s poses in %.3f seconds'%(len(detections)-1, time.time() - start_time))
        print('-------------')
        # save images with 2d bounding box for checking
        if FLAGS.check_detection:
            cv2.imwrite('./checking/2dbbox/' + img_id + '.png', img)
        # save prediction image for checking
        if FLAGS.check_projection:
            cv2.imwrite('./checking/BEV/' + img_id + '.png', check_img)
            


if __name__ == '__main__':
    main()
