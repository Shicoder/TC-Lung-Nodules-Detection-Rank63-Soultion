# --coding:utf-8 --
import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

random.seed(1321)
numpy.random.seed(1321)


def find_mhd_file(patient_id):
    for subject_no in range(settings.TRAIN_SUBSET_START_INDEX, 10):
        src_dir = settings.RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None



def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.VALIDATION_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    #归一化，体素间距为1，z保留512个最多
    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)


def process_pos_annotations_patient(src_path, patient_id):
    df_node = pandas.read_csv("../DSB2017/data/csv/train/annotations.csv")
    dst_dir = settings.TRAIN_EXTRACTED_IMAGE_DIR + "_labels/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_dir = dst_dir + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing /settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = helpers.load_patient_images(patient_id, settings.TRAIN_EXTRACTED_IMAGE_DIR, "*_i.png")

    pos_annos = []
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        diam_mm = annotation["diameter_mm"]
        print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(diam_mm, 2)))
        center_float = numpy.array([node_x, node_y, node_z])
        # center_int = numpy.rint((center_float-origin) / spacing)
        # center_int = numpy.rint((center_float - origin) )
        # print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        # patient_imgs=patient_imgs.swapaxes(0, 2)有问题
        # patient_imgs=patient_imgs.swapaxes(0, 1)
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(settings.TRAIN_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv", index=False)
    return [patient_id, spacing[0], spacing[1], spacing[2]]




def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(settings.VALIDATION_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.VALIDATION_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.VALIDATION_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.VALIDATION_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.VALIDATION_EXTRACTED_IMAGE_DIR)
        # os.mkdir(settings.VALIDATION_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(settings.VAL_SUBSET_START_INDEX, settings.VAL_SUBSET_TRAIN_NUM):
        src_dir = settings.RAW_SRC_DIR + "val_subset0" + str(subject_no) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")

        if only_process_patient is None and True:
            pool = multiprocessing.Pool(8)
            pool.map(process_image, src_paths)
        else:
            for src_path in src_paths:
                print(src_path)
                if only_process_patient is not None:
                    if only_process_patient not in src_path:
                        continue
                process_image(src_path)

if __name__ == "__main__":
    #归一化，生成mask图
    if True:
        only_process_patient = None
        process_images(delete_existing=False, only_process_patient=only_process_patient)

