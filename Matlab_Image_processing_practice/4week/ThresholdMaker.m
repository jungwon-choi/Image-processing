function [th] =  ThresholdMaker(img, N)
% Non-uniform 이미지의 양자화 Threshold 값을 자동으로 생성하는 함수 (자체 제작)
% img : 입력 이미지
% N    : 양자화 비트 수
    
th = [];                       % threshold 배열 초기화
th_maxnum = 2^N-1;  % threshold 배열의 최대 index

thresholdAdd(img, 0, 256, 1);   

    function [] = thresholdAdd(img_, low, high, pos)
        % threshold에 따라 이미지를 양자화하는 배열에 값을 추가하는 재귀함수 (트리 구조로 접근)
        
        % 현재 양자화 비트수 단계의 threshold 생성
        th(pos) = mean(img_(low <= img_ & img_ < high));
        
         % 다음 양자화 비트수 단계의 threshold 생성
        if 2*pos < th_maxnum
            thresholdAdd(img_, low,  th(pos), 2*pos);      
            thresholdAdd(img_, th(pos), high, 2*pos+1);
        end
        
    end
end
