# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Build CNN model using Tensorflow library / train model with face data set & test the performance of the model
'''
import cv2                      # OpenCV 라이브러리
import numpy as np              # Matrix 연산 라이브러리
import sys                      # 프로그램 입력사용을 위한 라이브러리
import timeit                   # 시간 측정을 위한 라이브러리

from os.path import join        # 파일 경로를 결합을 위한 라이브러리
from YOLO_00_constant import *  # 코드에 필요한 상수 추가

### 사용법 안내 함수 ###
def show_usage():
    print("[프로그램 사용법!]")
    print(">> python TermProject_02_CNN_trainNtest.py train\t: 데이터로부터 CNN 모델을 통해 학습")
    print(">> python TermProject_02_CNN_trainNtest.py test1\t: 테스트 데이터로 수행")
    print(">> python TermProject_02_CNN_trainNtest.py test1\t: Webcam 이미지로 수행")

### 프로그램 수행 Main 함수 ###
if __name__ == "__main__":

    ## 아무 인자도 입력하지 않은 경우
    if len(sys.argv) <= 1:
        show_usage()    # 사용법 안내
        exit()          # 프로그램 종료

    import YOLO_02_network_model    # YOLO CNN 모델 Class 라이브러리
    from YOLO_02_network_model import YOLO_CNN_EmotionRecognition, format_image, people_smile_predict

    network = YOLO_CNN_EmotionRecognition()     # 표정 인식 CNN 모델 Class 객체 생성

    ## 모델 학습을 선택한 경우
    if sys.argv[1] == 'train':
        network.start_training()    # 데이터로 부터 모델 학습 수행
        network.save_model()        # 학습된 모델 저장

    ## 테스트 데이터 통한 모델 테스트를 선택한 경우
    elif sys.argv[1] == 'test1':
        network.build_network()     # 학습된 모델 불러오기

        img_datas = np.load(join(DATASET_NPY_DIRECTORY, DATASET_IMAGES_FILENAME)).reshape([-1, SIZE_FACE, SIZE_FACE, 1]) # 48 x 48 이미지 matrix로 reshape
        img_labels = np.load(join(DATASET_NPY_DIRECTORY, DATASET_LABELS_FILENAME)).reshape([-1, len(EMOTIONS)])          # 레이블 데이터를 np array로 불러오기

        print("-> 학습된 Model test 수행")

        total_num = len(img_datas)
        k = 0
        true_num = 0
        for img_data, img_label in zip(img_datas,img_labels):
            k += 1
            frame = cv2.resize(img_data,(400,400))

            # 이미지를 학습된 모델로 판단하여 검출된 사람 수와 웃는 사람 수 반환
            people_no, smile_no, result_idx = people_smile_predict(network, frame, [img_data], [[0,0,48,48]])

            if people_no is not None:
                    #print("검출된 사람 수: {0}\t웃고 있는 사람: {1}\t{2}% smile".format(people_no, smile_no,smile_no/people_no*100))
                    #if smile_no is people_no :   # 검출된 사람과 웃고 있는 사람이 동일할 경우
                        # Shooting 판단
                        #cv2.putText(frame, "Shoot!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    if result_idx == np.argmax(img_label): true_num += 1

            #cv2.imshow("data test", frame)
            #if cv2.waitKey(1) & 0xFF == ord('q') :
            #    break
            sys.stdout.write("Process rate : ")
            sys.stdout.write("{0:.2f}".format(k/total_num*100))
            sys.stdout.write("%")
            sys.stdout.write("\r")
        print("-> 학습된 Model Accuracy : {0:.2f} %", true_num/total_num*100)

    ## Webcam을 통한 모델 테스트를 선택한 경우
    elif sys.argv[1] == 'test2':
        network.build_network()     # 학습된 모델 불러오기

        vidcap = cv2.VideoCapture(0)    # 카메라 스트림 연결
                                        # 0: 자체 카메라 1: USB 카메라

        # 카메라를 통해 학습된 모델 성능 테스트 수행
        while True: # 종료전까지 무한 반복

            ret, frame = vidcap.read()  # 카메라로 부터 프레임 받아오기

            if frame is None: # 받아온 프레임이 존재하지 않을 경우
                print("카메라가 존재하지 않거나 동작하지 않습니다!")
                break   # 프로그램 종료

            else:   # 프레임이 존재한 경우

                # 프레임으로 부터 얼굴을 검출하여 Input data 형식으로 변환
                faces_datas, faces_pos, _ = format_image(frame)

                # 이미지를 학습된 모델로 판단하여 검출된 사람 수와 웃는 사람 수 반환
                people_no, smile_no, _ = people_smile_predict(network, frame, faces_datas, faces_pos)

                if people_no is not None:
                    print("검출된 사람 수: {0}\t웃고 있는 사람: {1}\t{2}% smile".format(people_no, smile_no,smile_no/people_no*100))
                    if smile_no is people_no :   # 검출된 사람과 웃고 있는 사람이 동일할 경우
                        # Shooting 판단
                        cv2.putText(frame, "Shoot!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                cv2.imshow('Current frame', frame) # 판단된 현재 프레임 출력

                # 'q'를 누를경우 프로그램 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        vidcap.release()            # 카메라 스트림 해제
        cv2.destroyAllWindows()     # 이미지 창 모두 닫기

    else:
        show_usage()    # 사용법 안내
        exit()          # 프로그램 종료
