# from typing_extensions import final
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
target_dir = 'target'
data_dir = 'data'
results_dir = 'results'


def line_slope(line):
    # print(line)
    return (line[0][1]-line[1][1])/(line[0][0]-line[1][0]+1e-12)


def line_intercept(line):
    slope = line_slope(line)
    return line[0][1]-slope*line[0][0]


def point_to_line_distance(point, line):
    l_pt1, l_pt2 = line
    x, y = point
    l_slope = line_slope(line)
    l_c = line_intercept(line)
    p1, p2, p3 = np.array(l_pt1), np.array(l_pt2), np.array(point)
    # abs((x*l_slope)+y+l_c)/(math.sqrt(math.pow(l_slope, 2)+1e-12)+1e-12)
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)


def distance_between_lines(line1, line2):
    l1_pt1, l1_pt2 = line1
    l2_pt1, l2_pt2 = line2
    l1_slope, l2_slope = line_slope(line1), line_slope(line2)
    l1_c, l2_c = line_intercept(line1), line_intercept(line2)

    return abs(l2_c-l1_c)/(math.sqrt(math.pow(l1_slope, 2)+math.pow(l2_slope, 2)))


def getAngle(a, b, c):
    ba = a - b

    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def euclad_dist(p1, p2):
    import math
    return math.sqrt(math.pow((p1[0]-p2[0]), 2)+(math.pow((p1[1] - p2[1]), 2))+1e-10)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', cv2.resize(labeled_img, (512, 512)))
    # cv2.waitKey()
    return labeled_img


def imshow_stats(img, stats):
    img = img.copy()
    for (x, y, w, h, a) in stats:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), (2))
    cv2.imshow('bbox', img)
    return


def k_means_color_clustering(img, k_clusters=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    # print(pixel_values.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    k = k_clusters
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(gray.shape)

    return segmented_image, sorted(centers.tolist())


def draw_histogram(img):
    if(img.ndim > 2):
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(
        img.ravel(), 256, [0, 256])
    plt.plot(hist)
    plt.show()
    return


for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)

    segmented_image, _ = k_means_color_clustering(img, 2)

    ret, mask = cv2.threshold(segmented_image, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask)
    index_save = 0
    max_area = -1e9
    for ii, (x, y, w, h, a) in enumerate(stats):
        # print(a)
        if(ii == 0):
            continue
        if(a > max_area):
            max_area = a
            index_save = ii

    final_mask = np.uint8(labels == (index_save))*255

    contours = cv2.findContours(final_mask,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    wall_seg = np.zeros_like(final_mask)

    cv2.drawContours(wall_seg, contours, 0, (255), -1)
    img_wall = cv2.bitwise_and(img, img, mask=wall_seg)

    edges = cv2.Canny(img_wall, 50, 100)
    edges = cv2.bitwise_and(edges, edges, mask=wall_seg)

    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE,
                             np.ones((3, 3), np.uint8))

    edges = 255-edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE,
                             np.ones((7, 7), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE,
                             np.ones((1, 7), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE,
                             np.ones((7, 1), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges)
    index_save = 0
    max_area = -1e9

    for ii, ((x, y, w, h, a), center) in enumerate(zip(stats, centroids)):

        if(a > max_area):
            max_area = a
            index_save = ii

    final_mask = 255-np.uint8(labels == (index_save))*255

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE,
                                  np.ones((7, 7), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE,
                                  np.ones((7, 7), np.uint8))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        final_mask)
    # labelled_img = imshow_components(labels)
    # cv2.waitKey(0)
    final_img = np.zeros_like(img)

    final_img[:, :, 0] = final_mask.copy()
    final_img[:, :, 1] = final_mask.copy()
    final_img[:, :, 2] = final_mask.copy()
    final_img_ = np.zeros(
        shape=(final_img.shape[0]+20, final_img.shape[1]+20, final_img.shape[2]), dtype=final_img.dtype)
    final_mask_ = np.zeros(
        shape=(final_img.shape[0]+20, final_img.shape[1]+20), dtype=final_mask.dtype)
    final_img_[10:-10, 10:-10, :] = final_img
    final_mask_[10:-10, 10:-10] = final_mask

    # print(final_img_.shape)
    # # paints_bbox = []
    # # paints_centers = []
    # print(final_mask_.shape)
    # print(img.shape)
    # cv2.imshow("", cv2.resize(final_mask_, (512, 512)))
    # cv2.waitKey(0)
    paints_corners = []
    for ii, ((x, y, w, h, a), center) in enumerate(zip(stats, centroids)):
        if(ii == 0):
            continue
        ratio = a/(w*h)
        if(ratio > .7 and a > 5000):

            # print(a/(w*h), a)

            shape_crop = final_mask_[10+y-5:10+y+h+5, 10+x-5:10+x+w+5]
            # print(shape_crop.shape)
            # cv2.waitKey(0)
            object_index = (shape_crop > 0).nonzero()
            object_index = np.array([object_index[1], object_index[0]]).T
            object_extreme_points = cv2.convexHull(object_index)
            # print(object_extreme_points.shape)
            cv2.drawContours(shape_crop, [object_extreme_points], -1, 255, -1)
            shape_crop_img = final_img_[10+y-5: 10+y+h+5, 10+x-5: 10+x+w+5]
            shape_crop[shape_crop == 255] = 255

            shape_crop = cv2.morphologyEx(shape_crop, cv2.MORPH_CLOSE,
                                          np.ones((7, 7), np.uint8))
            shape = cv2.Canny(shape_crop, 50, 100)
            # cv2.imshow('test', cv2.resize(shape_crop, (512, 512)))

            cdst = shape_crop_img.copy()
            object_index = (shape_crop > 0).nonzero()
            object_index = np.array([object_index[1], object_index[0]]).T
            object_extreme_points = cv2.convexHull(object_index)
            inters_hull = cv2.convexHull(object_extreme_points)
            # print(inters_hull.shape)

            ii = 0
            points = [inters_hull[0]]
            for pt in inters_hull:
                if(euclad_dist(pt[0], points[-1][0]) > 30):
                    points.append(pt)

            points = np.array(points)
            last_angle = getAngle(
                points[-1][0], points[0][0], points[1][0])
            angles = [(last_angle, 0)]
            for ii in range(1, points.shape[0]):
                pt = points[ii]
                prev_pt = points[ii-1]
                if(ii+1 == points.shape[0]):
                    next_pt = points[0]
                else:
                    next_pt = points[ii+1]
                cur_angle = getAngle(prev_pt[0], pt[0], next_pt[0])
                angles.append((cur_angle, ii))

            angle_points = sorted(angles, key=lambda x: x[0])[:4]
            final_points = []
            for ii, (_, pt_index) in enumerate(angle_points):
                pt = points[pt_index]
                final_points.append((x+pt[0][0], y+pt[0][1]))
            paints_corners.append(final_points)
    target_names = os.listdir(target_dir)
    # cv2.imshow('dog', target_img)
    for corners in paints_corners:
        target_img = cv2.imread(os.path.join(
            target_dir, target_names[random.randint(0, len(target_names)-1)]))
        sort_cornres = sorted(corners, key=lambda x: x[1])

        [pt_1, pt_2, pt_3, pt_4] = sort_cornres
        top_left = pt_1 if(pt_1[0] < pt_2[0]) else pt_2
        top_right = pt_1 if(pt_1[0] > pt_2[0]) else pt_2
        bottom_left = pt_3 if(pt_3[0] < pt_4[0]) else pt_4
        bottom_right = pt_3 if(pt_3[0] > pt_4[0]) else pt_4
        dst_points, src_points = np.array([top_left, top_right, bottom_left, bottom_right], np.float32), np.array([[0, 0], [
            target_img.shape[1], 0], [0, target_img.shape[0]], [target_img.shape[1], target_img.shape[0]]], np.float32)
        # print(src_points)
        # print(dst_points)

        M = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
        # print(M)
        new_target = cv2.warpPerspective(
            target_img, M, dsize=(img.shape[1], img.shape[0]))

        ret, target_mask = cv2.threshold(
            (new_target[:, :, 0].copy()), 1, 255, cv2.THRESH_BINARY_INV)

        img = cv2.bitwise_and(img, img, mask=target_mask)
        img += new_target

    cv2.imwrite(os.path.join(results_dir, img_name), img)
    # cv2.waitKey(0)
    # print(paints_corners)
