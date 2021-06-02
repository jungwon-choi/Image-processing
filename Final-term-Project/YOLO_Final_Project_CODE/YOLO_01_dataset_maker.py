# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Auto CNN train face data set maker
'''
import cv2                       # OpenCV 라이브러리
#import copy                      #  깊은 복사하기 위한 라이브러리 (컬러 이미지 저장할 경우)
from itertools import chain     # 이미지 데이터를 string으로 변환하기 위한 라이브러리
import numpy as np               # Matrix 연산 라이브러리
from YOLO_00_constant import *  # 코드에 필요한 상수 라이브러리

## 레이블에 따른 표정상태 반환 ##
def label_num_to_str(x): return {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy(smile)', 4: 'sad', 5: 'surprised', 6: 'neutral'}[x]
## # 키입력에 따른 레이블 반환 ##
def label_key_to_num(x): return {ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4, ord('5'): 5, ord('6'): 6}.get(x, None)

face_cascade = cv2.CascadeClassifier(CASC_FACE_PATH)     # Haarcascade 분류 특징 값 (얼굴 정면)
eye_cascade = cv2.CascadeClassifier(CASC_EYE_PATH)       # Haarcascade 분류 특징 값 (눈)

label_num = 6        # 레이블 초기화 : 중립
file_count = 0       # 저장될 파일 index 초기화

dataset_labels = []  # 저장될 Dataset 레이블 리스트
dataset_images = []  # 저장될 Dataset 이미지 리스트

### 데이터 헤더 저장 과정 ###
f = open(join(DATASET_CSV_DIRECTORY,DATASET_CSV_FILENAME_MINE), "w")       # data set을 저장할 파일 스트림 연결
f.write("emotion,pixels,Usage\n")                   # 데이터 헤더 저장
f.close()                                               # 파일 스트림 연결 해제

vidcap = cv2.VideoCapture(0)    # 카메라 스트림 연결
                                # 0: 자체 카메라 1: USB 카메라

# 이미지를 개선하기 위해 사용한 contrast-limited adaptive histogram equlization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Contrast를 제한한 필터 생성

### 얼굴 검출 및 검출된 얼굴 이미지를 CNN 입력 네트워크 data size로 변환하여 csv파일 및 numpy 데이터로 저장  ###
while True:  # 종료전까지 무한 반복 저장

    ret, frame = vidcap.read()  # 카메라로 부터 프레임 받아오기

    if frame is None: # 받아온 프레임이 존재하지 않을 경우
        print("카메라가 존재하지 않거나 동작하지 않습니다!")
        break   # 프로그램 종료

    else:   # 프레임이 존재
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # 컬러 이미지를 흑백으로 변환 (연산량 감소를 위함)

        histequ_img = clahe.apply(gray)                             # contrast-limited adaptive histogram equlization 적용
        cv2.imshow("Comparison", np.hstack((gray,histequ_img)))    # 비교 프레임 출력
        gray = histequ_img

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # 흑백 이미지 내에서 얼굴 검출

        #frame_cr = copy.deepcopy(frame)                         # 컬러를 저장하기 위해 Draw할 이미지와 저장할 이미지를 따로 복사 (깊은 복사)

        # 모든 검출된 얼굴을 사각형으로 Boxing
        for (x,y,w,h) in faces: # 얼굴이 검출된 좌표 [(x,y) : 시작 좌표 / (w,h) : 폭, 높이]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 1)   # 직사각형으로 얼굴 영역 Draw (파란색)
            #cropped_face = frame_cr[y:(y + h), x:(x + w)]      # 얼굴 영역 Cropping (컬러)
            cropped_face_gray = gray[y:(y + h), x:(x + w)]      # 얼굴 영역 Cropping (흑백)

            eyes = eye_cascade.detectMultiScale(cropped_face_gray)  # 얼굴 내에서 눈 검출
            real_eyenum = 0                                         # 눈 개수 카운트 초기화

            # 눈 검출 ROI 설정을 위함
            mid_y = int((2*y + h) / 2.)              # 얼굴의 중심점 (y축)
            high_limit_ey = int(y + h * 0.22)        # 얼굴 눈 범위 상단 제한
            left_limit_ex = int(x + w * 0.15)        # 얼굴 눈 범위 좌측 제한
            right_limit_ex = int(x + w * 0.85)       # 얼굴 눈 범위 우측 제한

            # 직사각형으로 눈 ROI 영역 Draw (빨간색)
            cv2.rectangle(frame,(left_limit_ex,high_limit_ey),(right_limit_ex,mid_y),(0,0,255), 1)

            # 각 얼굴에 검출된 눈을 사각형으로 Boxing
            for (ex, ey, ew, eh) in eyes:   # 눈이 검출된 좌표 [(x,y) : 시작 좌표 / (w,h) : 폭, 높이]

                mid_ex = int((2*(x+ex) + ew)/2.)    # 눈 영역의 중심 x좌표
                mid_ey = int((2*(y+ey) + eh)/2.)    # 눈 영역의 중심 y좌표

                if mid_y > mid_ey and mid_ey > high_limit_ey : # 얼굴의 중심점과 이마 사이에 있고 (코, 입 등의 오검출 방지)
                                                                #  주의 : 아래쪽 위치한 픽셀의 좌표가 더 큼!
                    if left_limit_ex < mid_ex and mid_ex < right_limit_ex: # 미간 근처에 있을 경우 (안경 등의 오검출 방지)
                        real_eyenum += 1    # ROI 내의 눈 개수 카운트
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1) # 직사각형으로 눈 영역 Draw (초록색)


            if real_eyenum >= 2:    ## ROI 내에 객체가 2개 이상 검출된 경우 데이터 저장 수행 ##

                # Cropping한 얼굴을 Data set 크기에 맞게 resizing
                resized = cv2.resize(cropped_face_gray, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC)  # 흑백
                #resized = cv2.resize(frame_cr, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC)           # 컬러

                ## 이미지로 데이터 저장
                cv2.imshow("Saved Face Image", resized)                                           # 저장되는 얼굴 이미지 출력
                cv2.imwrite(join(DATASET_IMG_DIRECTORY,"dataImage%d.jpg") % file_count, resized)  # 이미지 .jpg 형식으로 저장
                print('Saved dataImage{0}.jpg'.format(file_count))                               # 이미지 저장 알림

                ## csv 파일로 데이터 저장
                f = open(join(DATASET_CSV_DIRECTORY,DATASET_CSV_FILENAME_MINE), "a")        # data set에 이어서 저장하기 위해 파일 스트림 연결
                image_data = str(list(chain.from_iterable(resized)))     # 이미지를 string 형태로 변환

                # 레이블, 이미지 데이터, 데이터 유형의 형식으로 string 데이터 생성
                data = ','.join([str(label_num),image_data.replace(',',' ').replace('[','').replace(']',''),'Traning\n'])

                f.write(data)   # data 기록
                f.close()       # 파일 스트림 연결 해제

                ## numpy 형태로 데이터 누적
                label_onehot = np.zeros(EMOTIONS_NUM)
                label_onehot[label_num] = 1.0

                dataset_labels.append(label_onehot)  # 레이블 데이터 누적
                dataset_images.append(resized)       # 이미지 데이터 누적

                file_count+=1   # 파일 index 증가

        cv2.imshow("Orignal frame", frame)        # 검출된 객체 Boxing된 현재 프레임 출력

        key =  cv2.waitKey(100)                      # 0.1초 동안 키입력 대기 (약 0.1초에 한프레임씩 저장)
        if key & 0xFF == ord('q'):                  # 'q'를 누를경우 프로그램 종료
            break
        elif key & 0xFF == ord('s') :               # 's'를 누를 경우 일시정지

            key_lb = cv2.waitKey(0) & 0xFF           # 변환할 레이블 선택 입력
            key_lb = label_key_to_num(key_lb)        # 입력받은 키를 레이블로 변환
            
            if key_lb is not None:                 # 정상적인 레이블인 경우
                label_num = key_lb                   # 해당 레이블로 저장 데이터 레이블 변경
            
            # 현재 레이블 알림
            print("current label: {0}".format(label_num_to_str(label_num)))

## numpy 형태로 데이터 저장
np.save(join(DATASET_NPY_DIRECTORY, DATASET_IMAGES_FILENAME_MINE), dataset_images)
np.save(join(DATASET_NPY_DIRECTORY, DATASET_LABELS_FILENAME_MINE), dataset_labels)

vidcap.release()            # 카메라 스트림 해제
cv2.destroyAllWindows()     # 이미지 창 모두 닫기
