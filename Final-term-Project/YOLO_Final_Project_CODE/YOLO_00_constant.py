# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : necessary constants for YOLO Code
'''
from os.path import isfile, join    # 파일의 존재 여부 확인과 경로를 결합하기 위한 라이브러리

# 각종 필요 파일 경로 및 이름 선언
CASC_FACE_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
CASC_EYE_PATH = './haarcascade_files/haarcascade_eye.xml'
DATASET_IMG_DIRECTORY = './data/Image/'
DATASET_CSV_DIRECTORY = './data/csv/'
DATASET_NPY_DIRECTORY = './data/npy/'
TRAINED_MODEL_DIRECTORY = './data/model/'
PHOTO_SAVE_DIRECTORY = './saved_photos/'

DATASET_CSV_FILENAME = 'fer2013.csv'
DATASET_IMAGES_FILENAME = 'data_images.npy'
DATASET_LABELS_FILENAME = 'data_labels.npy'
DATASET_CSV_FILENAME_MINE = 'facedata_set.csv'
DATASET_IMAGES_FILENAME_MINE = 'data_images_mine.npy'
DATASET_LABELS_FILENAME_MINE = 'data_labels_mine.npy'
TRAINED_MODEL_FILENAME = 'model_YOLO_CNN.tflearn'
TRAINED_MODEL_META_FILENAME = 'model_YOLO_CNN.tflearn.meta'

# 판단할 수 있는 표정들
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
# 구분 가능한 표정 종류의 수
EMOTIONS_NUM = len(EMOTIONS)

# Dataset의 이미지 크기(48 x 48)
SIZE_FACE = 48
