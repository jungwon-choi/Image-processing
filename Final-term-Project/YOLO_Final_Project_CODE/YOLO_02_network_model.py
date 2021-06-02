# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Build CNN model using Tensorflow library / train model with face data set & test the performance of the model
'''
# CNN 모델 구축 및 학습을 위한 Tensorflow 응용 라이브러리
import tflearn  # Tensorflow를 간단하게 사용하기 위한 MIT 오픈소스 라이브러리
from tflearn.layers.core import input_data, dropout, fully_connected           # DNN 모델링 함수들
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d              # Convolution과 pooling 함수
from tflearn.layers.merge_ops import merge                                     # Tensor list 를 하나로 합쳐주는 함수
from tflearn.layers.estimator import regression                                # gradient descent optimizer를 사용한 regression

# 데이터를 train과 test set으로 분리해주는 라이브러리
from sklearn.model_selection import train_test_split
from YOLO_00_constant import *  # 코드에 필요한 상수 라이브러리

import cv2               # OpenCV 라이브러리
import copy              # 깊은 복사하기 위한 라이브러리
import numpy as np       # Matrix 연산 라이브러리
import timeit            # 시간 측정을 위한 라이브러리

### 저장된 Dataset을 학습할 수 있는 형태로 Load 하는 Class ###
class DatasetLoader:
    def load_from_dataset(self):    # Dataset을 불러와 클레스 변수에 저장하는 함수

        images = np.load(join(DATASET_NPY_DIRECTORY, DATASET_IMAGES_FILENAME))  # 이미지 데이터를 np array로 불러오기
        images = images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])                  # 48 x 48 이미지 matrix로 reshape
        labels = np.load(join(DATASET_NPY_DIRECTORY, DATASET_LABELS_FILENAME)).reshape([-1, len(EMOTIONS)])    # 레이블 데이터를 np array로 불러오기

        # data set을 8:2 로 train과 test으로 분리
        self._images, self._images_test, self._labels, self._labels_test = train_test_split(images, labels, test_size=0.20, random_state=42)

    @property   # train images 호출
    def images(self):
        return self._images
    @property   # train labels 호출
    def labels(self):
        return self._labels
    @property   # test images 호출
    def images_test(self):
        return self._images_test
    @property   # test labels 호출
    def labels_test(self):
        return self._labels_test


### 저장된 Dataset을 학습할 수 있는 형태로 Load 하는 Class ###
class YOLO_CNN_EmotionRecognition:      # 표정 인식 CNN 모델 알고리즘 Class

    def __init__(self):                 # 객체 생성자
        self.dataset = DatasetLoader()  # DatasetLoader 객체를 클레스 변수에 선언

    ## CNN 모델 구축 함수 ##
    def build_network(self):
        # Smaller 'AlexNet' 모델 참고
        print("-> CNN 모델 생성중 ...",)
        self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])        # 이미지 입력 레이어 : 이미지 크기 48 x 48 x 1 (흑백)
        self.network = conv_2d(self.network, 64, 5, activation='relu')          # Convolution  연산 레이어 : 필터 개수 : 64, 필터 크기 : 5 x 5
        self.network = max_pool_2d(self.network, 3, strides=2)                   # Max Pooling 연산 레이어  : Pooling 크기 : 3 x 3, 수행 보폭 : 2
        self.network = conv_2d(self.network, 64, 5, activation='relu')          # Convolution  연산 레이어 : 필터 개수 : 64, 필터 크기 : 5 x 5
        self.network = max_pool_2d(self.network, 3, strides=2)                   # Max Pooling 연산 레이어  : Pooling 크기 : 3 x 3, 수행 보폭 : 2
        self.network = conv_2d(self.network, 128, 4, activation='relu')         # Convolution  연산 레이어 : 필터 개수 : 128, 필터 크기 : 4 x 4
        self.network = dropout(self.network, 0.3)                                # Overfitting 방지를 위한 dropout, 30% 랜덤하게 제거
        self.network = fully_connected(self.network, 3072, activation='relu')   # Fully Connected NN 레이어 : 인풋 : 3072 개
        self.network = fully_connected(                                          # Fully Connected NN 레이어 : 인풋 : 7 개 (표정 개수)
            self.network, EMOTIONS_NUM, activation='softmax')
        self.network = regression(self.network, optimizer='momentum', loss='categorical_crossentropy') # 학습 최적화 방법

        self.model = tflearn.DNN(                       # 네트워크 모델 저장
            self.network,                               # 생성한 모델 구조
            checkpoint_path=TRAINED_MODEL_DIRECTORY,    # 체크포인트 저장 위치
            max_checkpoints=1,                          # 체크포인트 개수
            tensorboard_verbose=2                       # Tensor Board 지원(Loss, Accuracy, Gradients, Weights.)
        )
        self.load_model()                               # 네트워크 모델 불러오기

    ## Dataset을 불러오는 함수 ##
    def load_saved_dataset(self):
        print("-> Dataset 로드중 ...",)
        self.dataset.load_from_dataset()

    ## Dataset을 통한 모델 학습 함수 ##
    def start_training(self):
        self.load_saved_dataset()        # Dataset 불러오기
        self.build_network()             # CNN 학습 모델 생성

        if self.dataset is None:        # Dataset이 존재하지 않은 경우
            self.load_saved_dataset()    # Dataset 불러오기

        print("-> Convolutional Neural Network 모델 학습중 ...")
        start_t = timeit.default_timer() # 시작시간 측정
        self.model.fit( # CNN 모델 학습
            self.dataset.images, self.dataset.labels,   # 학습 데이터
            validation_set=(self.dataset.images_test,   # 테스트 데이터
                            self.dataset.labels_test),
            n_epoch=100,                                # 100세대 학습
            batch_size=50,                              # 1세대 당 50개의 Dataset 사용
            shuffle=True,                              # Dataset 섞기
            show_metric=True,                          # 매 Step 마다 정확도 출력
            snapshot_step=200,                          # 200 Step마다
            snapshot_epoch=True,                       # 학습되고 있는 모델 저장
            run_id='YOLO_CNN_EmotionRecognition'
        )
        end_t = timeit.default_timer()  # 종료시간 측정
        print("학습 완료!\t걸린 시간 : {0:.3f}".format(end_t-start_t))

    ## 학습된 모델을 통해 얼굴의 표정을 예측 함수 ##
    def predict(self, images):
        if images is None:  # 얼굴 이미지가 존재하지 않을 경우
            return None     # None을 반환
        for i, image in enumerate(images):  # 얼굴 이미지들을 4차원 형태로 변환 (입력 레이어 4차원 Tensor)
            images[i] = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return list(map(self.model.predict, images))    # 각 이미지 별로 결과를 예측하여 결과를 리스트로 반환

    ## 학습된 모델을 저장하는 함수 ##
    def save_model(self):
        print("-> 학습된 모델 저장중 ...",)
        self.model.save(join(TRAINED_MODEL_DIRECTORY, TRAINED_MODEL_FILENAME))

    ## 학습된/학습할 모델을 불러오는 함수 ##
    def load_model(self):
        print("-> 학습 모델 로드중 ...")
        if isfile(join(TRAINED_MODEL_DIRECTORY, TRAINED_MODEL_META_FILENAME)):   # model의 meta 데이터가 존재할 경우
            self.model.load(join(TRAINED_MODEL_DIRECTORY, TRAINED_MODEL_FILENAME))  # model 정보 불러오기
        else:
          print("저장된 모델 파일을 찾을 수 없습니다.")

cascade_classifier = cv2.CascadeClassifier(CASC_FACE_PATH)   # 얼굴 검출을 위한 Haarcascade 모듈 (얼굴 정면)

### 원본 이미지에서 얼굴을 검출하여 검출된 얼굴 이미지를 입력 data 형식으로 변환하는 함수 ###
def format_image(frame):
    image = copy.deepcopy(frame)

    if len(image.shape) > 2 and image.shape[2] == 3:                 # 입력 이미지가 컬러인 경우
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               # 흑백 이미지로 변환
    #else:                                                            # 입력 이미지가 encoding된 흑백인 경우
    #    image = cv2.imdecode(image, 0)                                # decoding 수행 (0: 흑백 1: 컬러)

    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)  # 이미지에서 얼굴 검출

    if len(faces) is 0:                   # 얼굴을 검출 하지 못한 경우
        return [None, None, None]            # None을 반환

    faces_datas = []                     # 검출된 얼굴이미지를 담을 리스트
    faces_pos = []
    eye_open_no = 0
    for i, face in enumerate(faces):    # 각 얼굴에 대하여 Cropping 수행
        # 얼굴 위치 직사각형으로 draw (파란색)
        cv2.rectangle(frame, (face[0],face[1]), (face[0]+face[2],face[1]+face[3]), (255,0,0), 1)

        # 얼굴 영역 Cropping
        face_chop = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
        eye_open = eye_open_check(face_chop)
        if eye_open: eye_open_no += 1
        faces_pos.append(face)

        try:    #  네트워크 입력 크기에 맞게 이미지 Resizing
            faces_datas.append(cv2.resize(face_chop, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.)
        except Exception as err:    # Resizing 중에 Error가 발생한 경우
            print("이미지 resizing Error 발생 :", err)
            return [None, None, None] # None을 반환

    return [faces_datas, faces_pos, eye_open_no] # 이미지와 얼굴영역 위치 반환

### 이미지를 모델에 적용하여 웃음여부를 예측하여 사람 수를 반환하는 함수 ###
def people_smile_predict(network, frame, faces_datas, faces_pos):
    if faces_datas is not None:                  # 얼굴이 검출되었을 경우
        people_no = len(faces_datas)               # 검출된 사람 수 판단
        #start_t = timeit.default_timer()  # 시작시간 측정
        results = network.predict(faces_datas)     # 얼굴데이터를 통해 표정 예측
        #end_t = timeit.default_timer()  # 종료시간 측정
        #print("모델 예측 시간 : {0:.3f}".format(end_t-start_t))

        smile_no = 0                               # 웃고 있는 사람 수 초기화
        result_idx = None
        if len(results) is not 0:                 # 사람이 검출되었을 경우
            for i, result in enumerate(results):                # 각 사람의 판단 결과마다 반복
                result_idx = np.argmax(result[0])
                if result_idx == 3:   # 해당 사람이 Smile(3) 상태로 판단 되었을 경우
                    smile_no = smile_no + 1        # 웃고 있는 사람 수 카운트
                    cv2.putText(frame, "Smile", (faces_pos[i][0], faces_pos[i][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No Smile", (faces_pos[i][0], faces_pos[i][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return [people_no, smile_no, result_idx]
    return [None, None, None]

eye_cascade = cv2.CascadeClassifier(CASC_EYE_PATH)       # Haarcascade 분류 특징 값 (눈)

### 얼굴 이미지 리스트에서 눈을 뜨고 있는 사람 수를 판단하는 함수 ###
def eye_open_check(face_chop):
    if face_chop is not None:
        SIZE = 400
        face_data = cv2.resize(face_chop,(SIZE,SIZE))
        eyes = eye_cascade.detectMultiScale(face_data)   # 얼굴 내에서 눈 검출
        real_eyenum = 0                                  # 눈 개수 카운트 초기화

        # 눈 검출 ROI 설정을 위함
        mid_y = int(SIZE/2.)              # 얼굴의 중심점 (y축)
        high_limit_ey = int(SIZE * 0.22)        # 얼굴 눈 범위 상단 제한
        left_limit_ex = int(SIZE * 0.15)        # 얼굴 눈 범위 좌측 제한
        right_limit_ex = int(SIZE * 0.85)       # 얼굴 눈 범위 우측 제한

        # ROI내 눈 객체 개수 판단
        for (ex, ey, ew, eh) in eyes:   # 눈이 검출된 좌표 [(x,y) : 시작 좌표 / (w,h) : 폭, 높이]

            mid_ex = int((2*ex + ew)/2.)    # 눈 영역의 중심 x좌표
            mid_ey = int((2*ey + eh)/2.)    # 눈 영역의 중심 y좌표

            if mid_y > mid_ey and mid_ey > high_limit_ey : # 얼굴의 중심점과 이마 사이에 있고 (코, 입 등의 오검출 방지)
                #  주의 : 아래쪽 위치한 픽셀의 좌표가 더 큼!
                if left_limit_ex < mid_ex and mid_ex < right_limit_ex: # 미간 근처에 있을 경우 (안경 등의 오검출 방지)
                    real_eyenum += 1    # ROI 내의 눈 개수 카운트
                        
        if real_eyenum >= 2:    ## ROI 내에 객체가 2개 이상인 경우 ##
            return True
        else:
            return False
    return None
