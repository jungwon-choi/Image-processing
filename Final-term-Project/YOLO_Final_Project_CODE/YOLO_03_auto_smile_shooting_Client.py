# -*- coding: utf-8 -*-
'''
 Copyright ⓒ 2018 TEAM YOLO
 Video System Capstone Design
 Description : Camera Auto Smile Shooting System (Client part)
'''
import cv2              # OpenCV 라이브러리
import numpy as np      # 프레임을 Matrix로 연산하는 라이브러리
import socket           # 네트워크 소켓 라이브러리
import sys              # 프로그램 입력사용을 위한 라이브러리
import timeit           # 시간 측정을 위한 라이브러리
import datetime         # 현재 시간 정보를 얻기 위한 라이브러리

from os.path import join                             # 파일 경로를 결합을 위한 라이브러리
from YOLO_00_constant import PHOTO_SAVE_DIRECTORY    # 코드에 필요한 상수 추가


SERVER_TCP_IP = '127.0.0.1'   # 서버 IP주소 (본인 컴퓨터 로컬 IP주소)
SERVER_TCP_PORT = 2467          # 서버 포트번호

# 따로 입력이 있었을 경우 주소 수정
if len(sys.argv) > 1  :
    SERVER_TCP_IP = sys.argv[1]        # 서버 IP주소
    SERVER_TCP_PORT = sys.argv[2]      # 서버 포트번호

sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # 네트워크통신 TCP 소켓 생성
sock_tcp.connect((SERVER_TCP_IP,SERVER_TCP_PORT))               # 네트워크통신 TCP 소켓 연결 (서버에 연결)

vidcap = cv2.VideoCapture(0)   # 카메라 스트림 연결
                               # 0: 자체 카메라 1: USB 카메라

frame_cnt = 0   # 현재 프레임 번호
mode = 11   # Camera shooting Mode
            # 11 : Smille Auto shooting & Eye closed detection
            # 10 : Smille Auto shooting
            # 01 : Eye closed detection
            # 00 : Manual shooting

pass_key = 0         # 키 전달 조건 초기화
key = 0xFF           # 키 값 초기화

# 이미지를 개선하기 위해 사용한 contrast-limited adaptive histogram equlization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Contrast를 제한한 필터 생성

### 서버에 이미지 전송 및 서버로 부터 전송받은 결과에 따른 사진 저장 알고리즘 수행 ###
while True: # 종료전까지 무한 반복
    start_t = timeit.default_timer() # 시작시간 측정

    ret, frame = vidcap.read() # 카메라로 부터 프레임 받아오기

    if frame is None: # 받아온 프레임이 존재하지 않을 경우
        print("카메라가 존재하지 않거나 동작하지 않습니다!")
        break   # 프로그램 종료

    else:   # 프레임이 존재
        frame_cnt = frame_cnt + 1   # 현재 프레임 번호 갱신

        # 서버에 YOLO 알고리즘 옵션 전송
        sock_tcp.send(str(mode).encode())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환

        #histequ_img = cv2.equalizeHist(gray)                       # 이미지 개선을 위해 히스토그램 균일화 (Contrast 증가)
        #cv2.imshow("Comparison", np.hstack((gray,histequ_img)))    # 비교 프레임 출력

        histequ_img = clahe.apply(gray)                             # contrast-limited adaptive histogram equlization 적용
        #cv2.imshow("Comparison", np.hstack((gray,histequ_img)))    # 비교 프레임 출력

        transfer_img =  histequ_img  #gray       # 전송할 이미지 선택 (True : gray, False : histequ)

        # 전송량을 줄이기위해 JPEG 인코딩 수행 (90% quality)
        result, imgencode = cv2.imencode('.jpg', transfer_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # 전송을 위해 이미지 데이터를 바이트로 변환
        img_datas = imgencode.tobytes()

        # 전송할 이미지 데이터 크기 확인
        data_size = len(img_datas)

        # 서버로 이미지 데이터 크기 정보 전송
        sock_tcp.send(str(data_size).encode())

        pos = 0   # 전송중인 데이터의 현위치 (인덱스)
        buf_size = data_size     # 버퍼크기 (한번에 전송하도록 데이터 크기로 설정)
        #buf_size = 4096         # 쪼개서 여러번 보내는 것도 가능

        # 모든 데이터를 전송하기전까지 반복
        while pos < data_size:
            try:
                if pos + buf_size <= data_size:          # 다음에 전송되는 데이터의 위치가 마지막 위치를 초과하지 않을 경우
                    data = img_datas[pos:pos+buf_size]   # 이미지 버퍼 바이트만큼 읽음
                else:
                    data = img_datas[pos:data_size]      # 이미지 남은 바이트 모두 읽음

                sock_tcp.send(data)     # 서버로 이미지 데이터 전송
                pos = pos + buf_size    # 데이터 인덱스 업데이트

            except Exception as e:     # 전송중 오류가 발생했을 경우
                print(e)                # 오류 출력
                break                  # 프로그램 종료

        if mode == 0 : # Manual Shooting Mode인 경우
            pass_key = 1    # 이곳에서 키입력 받기
            key = cv2.waitKey(50)
            sock_tcp.send(str(key).encode())    # Server에 Manual shooting 신호 전송
        else:
            pass_key = 0    # 아래에서 키입력 받기

        temp = sock_tcp.recv(1024).decode() # 서버로 부터 판단 결과 수신
        if len(temp) == 0:
            result = 0      # "Not yet!"
        else:
            result = int(temp)   # 문자열 결과를 int형으로 변환

        if result == 1: # "Shoot!":
            cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.jpg')  # 현재 시간 측정
            cv2.imwrite(join(PHOTO_SAVE_DIRECTORY, cur_time), sv_frame)    # 이미지 .jpg 형식으로 저장


        end_t = timeit.default_timer()  # 종료시간 측정
        print("[프레임No.{0}]".format(frame_cnt), "  이미지크기: {0} -> {1}".format(histequ_img.size,data_size),"  걸린시간:", "{0:.3f}".format(end_t-start_t),'sec',"  촬영모드:",mode,"  판단결과:", result)

        sock_tcp.recv(1024).decode()    # 버퍼 비우기

        cv2.imshow("Video", frame)     # 현재 프레임 출력

        if pass_key == 0:
                key = cv2.waitKey(10)   # 프레임 출력 유지 및 키입력

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

sock_tcp.shutdown()   # 네트워크통신 소켓 연결 해제 및 종료
vidcap.release()      # 카메라 스트림 해제
