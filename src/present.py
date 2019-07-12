'''
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab
'''
import cv2
import pickle as pk
import numpy as np
import nibabel as nib
import os
import pandas as pd
click_point = (0, 0)
index = 0

def mouse_click(event, x, y, flags, param):
    '''
    Use for getting click points in real images
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    '''
    global click_point
    global index
    ann = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point  = (x, y)
        print("index: %d"%index)
        print("region:%d"%ann[index][click_point[1]][click_point[0]])

def get_region_by_click():
    f1 = open(".." + os.sep + "atlas_reference" + os.sep + "atlas_plot.pickle", 'rb')
    plot = pk.load(f1)
    f2 = open(".." + os.sep + "atlas_reference" + os.sep + "atlas_dict.pickle", 'rb')
    ann = pk.load(f2)
    global index

    cv2.namedWindow('show region')
    cv2.setMouseCallback('show region', mouse_click, param=[ann])

    for img_index in range(len(plot)):
        img = plot[index]
        cv2.imshow('show region', img)
        cv2.waitKey()
        index += 1

def show_ann_as_image(ann_frame):
    img_2_show = np.clip(ann_frame, 0, 255).astype(np.uint8)
    cv2.imshow('annotation', img_2_show)
    cv2.waitKey()

def show_img(img, name, introduction=False):
    '''
    Show the image in a proper size
    :param img: The image to show
    :param introduction: if true, will display instructions on image
    :return:
    '''
    img_show = img.copy()
    img_show = cv2.resize(img_show, (int(img_show.shape[1] * 0.5), int(img_show.shape[0] * 0.5)))
    if introduction:
        cv2.putText(img_show, name, (50, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'q: reduce weight', (50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'e: increase weight', (50, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'a: previous frame', (50, 150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 'd: next frame', (50, 200), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
        cv2.putText(img_show, 's: quit', (50, 250), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255))
    cv2.imshow('img_show', img_show)
    key = cv2.waitKey()
    return key

def load_img(img_dir, post_fix):
    if post_fix == 'nii':
        image = nib.load(img_dir)
        image = np.asarray(image.get_data()).astype(np.uint8)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)
    elif post_fix == 'npy':
        data = np.load(img_dir)
        label_list = np.unique(data).tolist()
        label_list.sort()
        index = 0
        for label in label_list:
            data[data == label] = index%255
            index += 1
        image = np.asarray(data).astype(np.uint8)
    else:
        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

def merge_layers(name, save_dir, postfix1='jpg', postfix2='tif', postfix3='npy'):
    '''
    Use for presenting single brain images
    :param img_dir: The folder holding the brain images
    :param root_dir: actually the root dir here means [root dir] + [folder of brain id]
    :param save_dir: The folder used for saving result given by save_main function. The present function will detect whether there are existing availabe result.
    :param plot_bead: if true, the result given by bead_finder will be presented on the image
    :return:
    '''
    length = len(os.listdir(os.path.join(save_dir, name, "bead")))
    index = 0
    weight = 0.5

    while(True):
        bead_dir = os.path.join(save_dir, name, "post_bead", "%d.%s"%(index, postfix1))
        tissue_dir = os.path.join(save_dir, name, "post_tissue", "%d.%s"%(index, postfix2))
        ann_dir = os.path.join(save_dir, name, "atlas", "%d.%s"%(index, postfix3))

        bead_frame = load_img(bead_dir, postfix1)
        transformed_frame = load_img(tissue_dir, postfix2)
        ann_frame = load_img(ann_dir, postfix3)

        img_frame_sum = cv2.addWeighted(bead_frame, 0.5, transformed_frame, 0.5, 0)
        img_frame_sum = cv2.addWeighted(ann_frame, weight, img_frame_sum, 1 - weight, 0)

        key = show_img(img_frame_sum, name, True)
        if key == 101:
            weight -= 0.01
            if weight < 0:
                weight = 1.
        elif key == 113:
            weight += 0.01
            if weight > 1:
                weight = 0.
        elif key == 115:
            break
        elif key == 97:
            index += 1
            if index >= length:
                index = 0
        elif key == 100:
            index -= 1
            if index < 0:
                index = length - 1

def plot_beads_on_tissue(img_dir, csv_dir):
    warped_list = []
    for warped in os.listdir(img_dir):
        warped_list.append(warped)

    for w in warped_list:
        key = int(w.replace('.tif', ""))
        temp_csv_dir = os.path.join(csv_dir, "%d.csv"%key)
        points = []
        if os.path.exists(temp_csv_dir):
            df = pd.read_csv(temp_csv_dir)
            print(df['x'].shape)
            for i in range(df['x'].shape[0]):
                points.append((int(df['y'][i]), int(df['x'][i])))

        img = cv2.imread(os.path.join(img_dir, w))
        img = np.asarray(img).astype(np.uint8)
        for point in points:
            cv2.circle(img, point, 5, (255, 0, 0), thickness=2)
        cv2.imshow('1', img)
        cv2.waitKey()


if __name__ == '__main__':
    # plot_beads_on_tissue('/home/silasi/ants_data/66148-2/tissue', '/home/silasi/ants_data/66148-2/bead')
    load_img("/home/silasi/ants_data/66148-2/ann/123.npy", "npy")