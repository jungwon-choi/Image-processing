function [qt_img, Y_qtfunc, th] = Quantizaion_nuni(img, X, N)
% Non-uniform 이미지의 양자화 Threshold 값을 자동으로 생성하여 Decoder Mapping하는 함수 (자체 제작)
% ----------INPUT 데이터----------
% img              : 입력 이미지
% X                 : 이미지 픽셀 값 범위
% N                 : 양자화 비트 수
% ----------OUTPUT 데이터----------
% qt_img        : 양자화된 이미지 (Non-Uniform)
% Y_qtfunc     : 양자화 함수 (Non-Uniform)
% th               : 생성된 Threshold
% ---------------------------------
[H, W] = size(img);                         % 이미지의 크기 추출

th = [];                                          % threshold 배열 초기화
qt_img = zeros(H,W);                      % Quantization될 이미지 공간 생성
Y_qtfunc = zeros(1, length(X));       % N-bit Non-Uniform Quantization Function [Y = Q(X)]

th_maxnum = 2^N-1;                      % threshold 배열의 최대 index

QuantizatioinAdd(img, 0, 256, 1);      % 이미지 양자화 수행

    function [] = QuantizatioinAdd(img_, low, high, pos)
        % threshold 배열에 값에 따라 이미지를 Decoder Mapping하는 재귀함수 (트리 구조로 접근)
        
        % 현재 양자화 비트수 단계의 threshold 생성
        th(pos) = mean(img_(low <= img_ & img_ < high));
        
        % 현재 양자화 비트수 단계의 threshold에 따른 이미지 Decoder Mapping
         qt_img(low <= img_ & img_ < th(pos)) = (low+th(pos))/2;
         qt_img(th(pos) <= img_ & img_ < high) = (th(pos)+high)/2;
         
         % 현재 양자화 비트수 단계의 threshold에 따른 X에 대한 Quantization Function 생성
         Y_qtfunc(low <= X & X < th(pos)) = (low+th(pos))/2;
         Y_qtfunc(th(pos) <= X & X < high) = (th(pos)+high)/2;
         
         % 다음 양자화 비트수 단계로 이동
        if 2*pos < th_maxnum
            QuantizatioinAdd(img_, low,  th(pos), 2*pos);
            QuantizatioinAdd(img_, th(pos), high, 2*pos+1);
        end
    end    
end