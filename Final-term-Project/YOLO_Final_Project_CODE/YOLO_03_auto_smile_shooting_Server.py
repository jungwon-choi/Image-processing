# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Camera Auto Smile Shooting System (Client part)
'''
import cv2          # OpenCV 라이브러리
import numpy as np  # 프레임을 Matrix로 연산하는 라이브러리
import socket       # 네트워크 소켓 라이브러리
import threading    # 멀티 클라이언트를 위한 멀티스레딩 라이브러리
import sys          # 프로그램 입력사용을 위한 라이브러리
import timeit       # 시간 측정을 위한 라이브러리

import YOLO_02_network_model    # YOLO CNN 모델 Class 라이브러리
from YOLO_02_network_model import YOLO_CNN_EmotionRecognition, format_image, people_smile_predict, eye_open_check
from YOLO_00_constant import CASC_FACE_PATH  # 코드에 필요한 상수 추가

SERVER_TCP_IP = '127.0.0.1'   # 서버 IP주소 (본인 컴퓨터 로컬 IP주소)
SERVER_TCP_PORT = 2467          # 서버 포트번호

# Auto Smile Shooting YOLO Algorithm
def YOLO():
    client_socket, addr = server_sock_tcp.accept()  # 클라이언트 연결 수락
    tread = threading.Thread(target = YOLO)         # 멀티스레딩 초기화
    tread.start()                                   # 멀티스레딩 동작 시작

    # 연결된 클라이언트 정보 출력
    print("클라이언트가 연결되었습니다! ", addr[0], addr[1])

    frame_cnt = 0   # 현재 프레임 번호
    mode = 11   # Camera shooting Mode
                # 11 : Smille Auto shooting & Eye closed detection
                # 10 : Smille Auto shooting
                # 01 : Eye closed detection
                # 00 : Manual shooting

    people_no_max = 0    # 최대 검출 인원수 초기화
    captured = 0         # 1 cycle 내의 촬영 여부

    ### 클라이언트로 부터 이미지 수신 및 Shooting 여부 판단 결과를 클라이언트에게 전송하는 알고리즘 수행 ###
    while True: # 종료전까지 무한 반복
        try:
            start_t = timeit.default_timer() # 시작시간 측정
            frame_cnt = frame_cnt + 1   # 현재 프레임 번호 갱신

            # 클라이언트로부터 YOLO 알고리즘 모드 수신
            mode = int(client_socket.recv(1024).decode())

            # 전송될 이미지 크기 수신
            data_size = int(client_socket.recv(1024).decode())

            buf_size = data_size    # 버퍼크기 (한번에 전송하도록 이미지 크기로 설정)
            #buf_size = 4096        # 쪼개서 여러번 받는 것도 가능

            transfered_datas = b''  # 전송받은 데이터 초기화
            while len(transfered_datas) < data_size:  # 이미지를 전부 전송받을 때까지 반복
                data = client_socket.recv(buf_size)    # 클라이언트로 부터 이미지 데이터 수신
                transfered_datas += data               # 이미지 데이터 누적

            frame_data = np.frombuffer(transfered_datas ,dtype = np.uint8)  # 바이트를 다시 정수형으로 변환
            frame = cv2.imdecode(frame_data, 0)       # 이미지 프레임으로 JPEG 디코딩 (0: 흑백 1: 컬러)

            result = 0 # "Not yet!"
            if mode != 0:
                # 프레임으로 부터 얼굴을 검출하여 Input data 형식으로 변환
                faces_datas, faces_pos, eye_open_no = format_image(frame)

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
                                print(sm_check,eye_check, people_no, eye_open_no, people_no_max)
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
                key = int(client_socket.recv(1024).decode())    # Client로 부터 Manual shooting 신호 수신
                if key & 0xFF == ord('s'):  # 수동으로 사진촬영
                    result = 1  # "Shoot!"

            # 사람 수 보정 없는 코드
            '''
            people_no, smile_no = people_smile_predict(network, frame, faces_datas, faces_pos)

            result = 0 #"Not yet!"
            if people_no is not None:           # 검출된 사람이 존재하면서
                 if smile_no is people_no :   # 최대로 검출된 사람과 웃고 있는 사람이 동일할 경우
                    result = 1 # "Shoot!"
          '''

            client_socket.send(str(result).encode()) # 클라이언트에게 판단 결과 전송

            if result == 1: # "Shoot!"
                cv2.putText(frame, "Shoot!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            end_t = timeit.default_timer()  # 종료시간 측정
            print("{0} {1} : [프레임No.{2}]".format(addr[0], addr[1], frame_cnt), "  이미지크기: {0} -> {1}".format(data_size,frame.size),"  걸린시간:", "{0:.3f}".format(end_t-start_t),'sec',"  촬영모드:",mode,"  판단결과:", result)


            cv2.imshow('transfered image', frame)  # 전송 받은 이미지 출력

            # 'q'를 누를 경우 프로그램 종료
            if cv2.waitKey(1) & 0xFF == ord ('q'):
                global power_sig    # 전역 변수
                power_sig = 0        # 서버 종료
                break               # 프로그램 종료

            client_socket.send(' '.encode())    # 버퍼를 비우기 위해 공백 전송

        except Exception as e:  # 동작중 오류가 발생한 경우
            print("클라이언트가 종료되었습니다! ", addr[0], addr[1])
            #print("Error Type: ", e)
            cv2.destroyAllWindows() # 프레임 창 닫기
            break                  # 알고리즘 동작 종료

    client_socket.close()   # 클라이언트 연결 해제

# 따로 입력이 있었을 경우 주소 수정
if len(sys.argv) > 1 :
    SERVER_TCP_IP = sys.argv[1]        # 서버 IP주소
    SERVER_TCP_PORT = sys.argv[2]      # 서버 포트번호

# 예측 모델 불러오기
network = YOLO_CNN_EmotionRecognition()     # 표정 인식 CNN 모델 Class 객체 생성
network.build_network()                     # 학습된 모델 불러오기
cascade_classifier = cv2.CascadeClassifier(CASC_FACE_PATH)   # 얼굴 검출을 위한 Haarcascade 모듈 (얼굴 정면)

server_sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)     # 네트워크통신 TCP 소켓 생성
server_sock_tcp.bind((SERVER_TCP_IP, SERVER_TCP_PORT))                  # 서버 소켓에 IP주소와 포트 번호 연결
server_sock_tcp.listen(0)                                               # 클라이언트 연결 대기

power_sig = 1   # 서버 동작 신호
print("-> [서버 동작중] IP ADDR : {0} PORT NUM : {1}".format(SERVER_TCP_IP, SERVER_TCP_PORT))
while power_sig == 1: YOLO()  # Auto Smile Shooting YOLO Algorithm
server_sock_tcp.close()        # 네트워크통신 서버 소켓 연결 해제
