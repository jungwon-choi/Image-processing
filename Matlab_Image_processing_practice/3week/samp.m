function [img_sampled] =  samp(img, n, method)
%----------------------------------------------------------------------
% samp : 이미지를 원하는 LPF로 Sampling 을 수행하는 함수
%
% img : 입력받은 이미지
% n : 샘플링 간격 (이미지는 1/n 배)
% method : 샘플링을 수행할 방법 (LPF 선택)
%----------------------------------------------------------------------

% 입력받은 옵션에 따라 이미지 샘플링 방법 선택
if strcmp(method, 'Simple')
    key = 0;
elseif strcmp(method, 'Uniform')
    key = 1;
elseif strcmp(method, 'Gaussian')
    key = 2;
else  % 입력받은 옵션이 존재하지 않은 경우
    img_sampled = 'NULL';
    disp("아래의 Sampling 방법을 선택해주세요.")
    disp("method : 'Simple'(no LPF), 'Uniform', 'Gaussian' ")
    return
end

[H, W, D] = size(img);
img = double(img);

switch key
    
    case 0 % LPF 필터를 적용없이 샘플링
        img_sampled = img(1:n:end, 1:n:end,:); % n 간격으로 샘플링
        
    case 1  % Uniform Weight Moving average LPF 필터를 적용하여 샘플링
        lpf = ones(3,3,3)/9;        % Uniform LPF 필터
        img_lpf = zeros(H,W,D);  % LPF 적용할 이미지 공간 초기화
        
        for i = 1:H-2
            for j = 1:W-2
                img_lpf(i+1,j+1,:) = sum(sum( img(i:i+2,j:j+2,:).*lpf, 1),2);
                % .곱을 통하여 3x3필터를 적용하여 배열의 합을 현재 픽셀 값으로 저장
                % 테두리 픽셀의 경우 필터를 적용할 외부의 값이 없기 때문에 테두리 안쪽 범위 공간만 필터 적용
            end
        end

        img_sampled = img_lpf(1:n:end, 1:n:end,:); % n 간격으로 샘플링
        
    case 2 % Gaussian LPF 필터를 적용하여 샘플링
        lpf = [1,2,1; ...
                 2,4,2; ...
                 1,2,1]/16;             % Gaussian LPF 필터
        img_lpf = zeros(H,W,D);  % LPF 적용할 이미지 공간 초기화
  
        
        for i = 1:H-2
            for j = 1:W-2
                img_lpf(i+1,j+1,:) = sum(sum( img(i:i+2,j:j+2,:).*lpf, 1),2);
                % .곱을 통하여 3x3필터를 적용하여 배열의 합을 현재 픽셀 값으로 저장
                % 테두리 픽셀의 경우 필터를 적용할 외부의 값이 없기 때문에 테두리 안쪽 범위 공간만 필터 적용
            end
        end
        
        img_sampled = img_lpf(1:n:end, 1:n:end,:); % n 간격으로 샘플링 
        
    otherwise
        warning('이 경우는 없을듯?!')
        img_sampled = img;
        return
end

end