# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Camera Auto Smile Shooting System (All in one)
'''
import cv2              # OpenCV 라이브러리
import numpy as np      # 프레임을 Matrix로 연산하는 라이브러리
import sys              # 프로그램 입력사용을 위한 라이브러리
import timeit           # 시간 측정을 위한 라이브러리
import copy             #  깊은 복사하기 위한 라이브러리
import datetime         # 현재 시간 정보를 얻기 위한 라이브러리

from os.path import join                             # 파일 경로를 결합을 위한 라이브러리
from YOLO_00_constant import PHOTO_SAVE_DIRECTORY    # 코드에 필요한 상수 추가

import YOLO_02_network_model    # YOLO CNN 모델 Class 라이브러리
from YOLO_02_network_model import YOLO_CNN_EmotionRecognition, format_image, people_smile_predict, eye_open_check
from YOLO_00_constant import CASC_FACE_PATH, PHOTO_SAVE_DIRECTORY  # 코드에 필요한 상수 추가


# 예측 모델 불러오기
network = YOLO_CNN_EmotionRecognition()     # 표정 인식 CNN 모델 Class 객체 생성
network.build_network()                     # 학습된 모델 불러오기
cascade_classifier = cv2.CascadeClassifier(CASC_FACE_PATH)   # 얼굴 검출을 위한 Haarcascade 모듈 (얼굴 정면)
# 이미지를 개선하기 위해 사용한 contrast-limited adaptive histogram equlization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Contrast를 제한한 필터 생성

# Auto Smile Shooting YOLO Algorithm

vidcap = cv2.VideoCapture(0)   # 카메라 스트림 연결
                               # 0: 자체 카메라 1: USB 카메라

frame_cnt = 0   # 현재 프레임 번호
mode = 11   # Camera shooting Mode
            # 11 : Smille Auto shooting & Eye closed detection
            # 10 : Smille Auto shooting
            # 01 : Eye closed detection
            # 00 : Manual shooting

people_no_max = 0    # 최대 검출 인원수 초기화
captured = 0         # 1 cycle 내의 촬영 여부
pass_key = 0         # 키 전달 조건 초기화
key = 0xFF           # 키 값 초기화

### 클라이언트로 부터 이미지 수신 및 Shooting 여부 판단 결과를 클라이언트에게 전송하는 알고리즘 수행 ###
while True: # 종료전까지 무한 반복
        start_t = timeit.default_timer() # 시작시간 측정

        ret, frame = vidcap.read() # 카메라로 부터 프레임 받아오기

        if frame is None: # 받아온 프레임이 존재하지 않을 경우
            print("카메라가 존재하지 않거나 동작하지 않습니다!")
            break   # 프로그램 종료

        else:   # 프레임이 존재
            frame_cnt = frame_cnt + 1   # 현재 프레임 번호 갱신
            smile_no = 0         # 웃는 사람 수 초기화
            eye_open_no = 0      # 눈뜬 사람 수 초기화

            sv_frame = copy.deepcopy(frame)                 # 저장할 프레임 깊은 복사

            result = 0 #"Not yet!"
            if mode != 0:
                pass_key = 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환

                #histequ_img = cv2.equalizeHist(gray)                       # 이미지 개선을 위해 히스토그램 균일화 (Contrast 증가)
                #cv2.imshow("Comparison", np.hstack((gray,histequ_img)))    # 비교 프레임 출력

                histequ_img = clahe.apply(gray)                             # contrast-limited adaptive histogram equlization 적용
                #cv2.imshow("Comparison", np.hstack((gray,histequ_img)))    # 비교 프레임 출력

                # 프레임으로 부터 얼굴을 검출하여 Input data 형식으로 변환
                faces_datas, faces_pos, eye_open_no = format_image(histequ_img)

                # 인원수 오판단 보정
                if frame_cnt % 10 < 6:   # 초반 6개 프레임은 사람 인원수 카운트만
                    if frame_cnt % 10 == 0: people_no_max = captured = 0    # 최대 검출 인원수 및 촬영 여부 초기화

                    if faces_datas is not None:
                        people_no = len(faces_datas)
                        people_no_max = max(people_no_max,people_no)    # 인식된 사람 수가 최대인 경우 체크
                        cv2.putText(frame, "People : "+str(people_no_max), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

                else:  # 후반 3개 프레임은 사람 수 카운트 + Smile 인식 + 눈 감음 인식
                    if captured == 0:
                        # 이미지를 학습된 모델로 판단하여 검출된 사람 수와 웃는 사람 수 반환
                        if faces_datas is not None:
                            people_no = len(faces_datas)
                            people_no_max = max(people_no_max, people_no)    # 인식된 사람 수가 최대인 경우 체크

                            eye_check = sm_check = False
                            if mode - 10 >= 0:  # Smile Auto Shooting 기능이 활성화 된 경우
                                people_no, smile_no, _ = people_smile_predict(network, frame, faces_datas, faces_pos)
                                sm_check = smile_no == people_no_max

                            if mode % 10 == 1: # Eye closed detection 기능이 활성화 된 경우
                                eye_check = eye_open_no == people_no_max

                            cv2.putText(frame, "People : "+str(people_no_max), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

                            # 모드에 따른 결과 판단
                            if people_no is not None:       # 검출된 사람이 존재하면서
                                #print(sm_check,eye_check, people_no, eye_open_no, people_no_max)
                                if mode == 10 and sm_check :   # 최대로 검출된 사람 수와 웃고 있는 사람 수가 동일할 경우
                                    result = 1 # "Shoot!"
                                    captured = 1
                                elif mode == 1 and eye_check :   # 최대로 검출된 사람 수와 눈을 뜬 사람 수가 동일할 경우
                                    result = 1 # "Shoot!"
                                    captured = 1
                                elif mode == 11 and sm_check & eye_check: # 최대로 검출된 사람 수와 웃고 있는 사람 수, 눈을 뜬 사람 수가 모두 동일 한 경우
                                    result = 1 # "Shoot!"
                                    captured = 1
            else:
                pass_key = 1            # 이곳에서 키입력 받기
                key = cv2.waitKey(50)   # Manual shooting 신호 수신
                if key & 0xFF == ord('s'):  # 수동으로 사진촬영
                    result = 1  # "Shoot!"

            if result == 1: # "Shoot!":
                cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.jpg')   # 현재 시간 측정
                cv2.putText(frame, "Shoot!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imwrite(join(PHOTO_SAVE_DIRECTORY, cur_time), sv_frame)  # 이미지 .jpg 형식으로 저장

            end_t = timeit.default_timer()  # 종료시간 측정
            print("[프레임No.{0}]".format(frame_cnt), "걸린시간:", "{0:.3f}".format(end_t-start_t),'sec',"  촬영모드:",mode,"  판단결과:", result,
                  "\t최대 인원 수: {0}\t웃는 사람: {1}\t눈 뜬 사람: {2}".format(people_no_max, smile_no, eye_open_no))

            cv2.imshow('current image', frame)  # 이미지 출력
            if pass_key == 0:
                key = cv2.waitKey(10)             # 프레임 출력 유지 및 키입력

             # 'p'를 누를 경우 선택 옵션
            if key & 0xFF == ord('p'):
                key = cv2.waitKey(0)    # 옵션 선택 대기
                # 'q'를 누를 경우 프로그램 종료
                if key & 0xFF == ord('q'):
                    break   # 프로그램 종료
                # 's'를 누를 경우 smile auto shooting mode toggle
                elif key & 0xFF == ord('s'):
                    if mode < 10: mode +=10
                    else: mode -=10
                    print("촬영 모드가 변경되었습니다! Mode: ", mode)
                # 'e'를 누를 경우 eye detection mode toggle
                elif key & 0xFF == ord('e'):
                    if mode % 2 == 0: mode +=1
                    else: mode -=1
                    print("촬영 모드가 변경되었습니다! Mode: ", mode)

vidcap.release()      # 카메라 스트림 해제
cv2.destroyAllWindows() # 프레임 창 닫기
