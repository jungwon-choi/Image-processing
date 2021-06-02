%% 4week HW Uniform & Non-uniform Quantization
%    Copyright 2018, Jungwon Choi, INHA Electronics
close all; clear;

img = imread('fig2.jpg');   % 원하는 이미지를 불러오기
img = rgb2gray(img);        % 이미지를 흑백으로 변환
img = double(img);            % 연산을 위해 픽셀 값을 실수형으로 변환

[H, W] = size(img);            % 이미지의 크기 추출
N = 7;                               % 확인할 양자화 Bits (1~N bit) 
%% Uniform Quantization

img_nbit_uni = zeros(H,W,N);    % 각 Bitrate로 Uniform Quantization된 이미지를 저장할 공간
X = 1 : 255;                              % Quantization Function의 X축
Y_uni = zeros(length(X),N);       % 각 Bitrate에 따른 Quantization Function [Y = Q(X)]
err_nbit_uni = zeros(1,N);           % 각 Bitrate에 따른 Quantization Error (RMSE)

for Nbit = 1:N  % 1~N Bitrate로 Quantizaion 
    img_temp = zeros(H,W);  % 연산을 위한 임시 이미지 공간 생성
    for k = 1 : 2^Nbit % Quantizaion Step
        img_temp(round((256/2^Nbit)*(k-1)) <= img & img < round((256/2^Nbit)*k)) = round((256/2^Nbit)*(2*k-1)/2);
        % 조건식으로 Encoder Mapping 한 뒤 양 boundary의 중간 값을 대입하여 Decoder Mapping
        % 최대 픽셀값인 255를 양자화 범위에 포함하기 위하여 조건의 범위를 img < 256으로 설정
        Y_uni(round((256/2^Nbit)*(k-1)) <= X & X < round((256/2^Nbit)*k), Nbit) = round((256/2^Nbit)*(2*k-1)/2);
        % 해당 Bitrate일 때의 X에 대한 Quantization Function 생성
    end
    img_nbit_uni(:,:,Nbit) = img_temp;  % 해당 Bitrate로 Uniform Quantization를 적용한 이미지 결과 저장
    err_nbit_uni(1,Nbit) = sqrt(sum(sum((img-img_nbit_uni(:,:,Nbit)).^2))/length(img(:))); % Quantization Error (RMSE) 산출
end

% Bitrate에 따른 Quantization Function 출력
figure(50); set(50, 'name', ['Uniform Quantization Function 1-',num2str(N),'bit'],'units','normalized','outerposition', [0 0 1 1]);
for Nbit = 1:N 
    subplot(2,4,Nbit); plot(X, Y_uni(:,Nbit)); xlim([0 255]); ylim([0 255]); title([num2str(Nbit),'-bit Uniform Quantizaion Function']); 
end
subplot(2,4,N+1); histogram(img(:),length(X),'Normalization','probability');
title('Histogram of image'); xlabel('Pixel Value'); ylabel('Numer of Value (%)'); xlim([0 255]);

% Bitrate에 따른 Quantization Image 출력
figure(51); set(51, 'name', ['Uniform Quantization Image 1-',num2str(N),'bit'],'units','normalized','outerposition', [0 0 1 1]);
for Nbit = 1:N 
    subplot(2,4,Nbit); imshow(uint8(img_nbit_uni(:,:,Nbit))); title([num2str(Nbit),'-bit Uniform Quantizaion Image']); 
    xlabel(['Quantization ERR (RMSE) = ', num2str(err_nbit_uni(Nbit))]);
end
subplot(2,4,N+1); imshow(uint8(img)); title('Original Image'); 

%% Non-uniform Quantization (평균값을 이용)

% 이미지와 Bitrate에 따라 Threshold 값을 자동으로 생성하는 함수 (자체 제작)

img_nbit_nuni = zeros(H,W,N);    % 각 Bitrate로 Non-uniform Quantization된 이미지를 저장할 공간
X = 1 : 255;                                % Quantization Function의 X축
Y_nuni = zeros(length(X),N);       % 각 Bitrate에 따른 Quantization Function [Y = Q(X)]
err_nbit_nuni = zeros(1,N);           % 각 Bitrate에 따른 Quantization Error (RMSE)

for Nbit = 1:N  % 1~N Bitrate로 Quantizaion 
    [img_nbit_nuni(:,:,Nbit), Y_nuni(:, Nbit), th] = Quantizaion_nuni(img, X, Nbit);  % 자체 제작 Non-uniform Quantization 함수
    % 해당 Bitrate로 Non-uniform Quantization를 적용한 이미지와 Quatization Function 결과 저장
    err_nbit_nuni(1,Nbit) = sqrt(sum(sum((img-img_nbit_nuni(:,:,Nbit)).^2))/length(img(:))); % Quantization Error (RMSE) 산출
end

% Bitrate에 따른 Quantization Function 출력
figure(52); set(52, 'name', ['Non-uniform Quantization Function 1-',num2str(N),'bit'],'units','normalized','outerposition', [0 0 1 1]);
for Nbit = 1:N 
    subplot(2,4,Nbit); plot(X, Y_nuni(:,Nbit)); xlim([0 255]); ylim([0 255]); title([num2str(Nbit),'-bit Non-uniform Quantizaion Function']); 
end
subplot(2,4,N+1); histogram(img(:),length(X),'Normalization','probability'); % 원본 이미지 히스토그램 출력
title('Histogram of Image'); xlabel('Pixel Value'); ylabel('Numer of Value (%)'); xlim([0 255]);

% Bitrate에 따른 Quantization Image 출력
figure(53); set(53, 'name', ['Non-uniform Quantization Image 1-',num2str(N),'bit'],'units','normalized','outerposition', [0 0 1 1]);
for Nbit = 1:N 
    subplot(2,4,Nbit); imshow(uint8(img_nbit_nuni(:,:,Nbit))); title([num2str(Nbit),'-bit Non-uniform Quantizaion Image']); 
    xlabel(['Quantization ERR (RMSE) = ', num2str(err_nbit_nuni(Nbit))]);
end
subplot(2,4,N+1); imshow(uint8(img)); title('Original Image'); % 원본 이미지 출력

%%
% Bitrate에 따른 Quantization Image 출력
figure(55); set(55, 'name', 'Uniform & Non-uniform Quantization Image 1-3bit','units','normalized','outerposition', [0 0 1 1]);
for Nbit = 1:3 
    subplot(2,4,Nbit); imshow(uint8(img_nbit_uni(:,:,Nbit))); title([num2str(Nbit),'-bit Uniform Quantizaion Image']); 
    xlabel(['Quantization ERR (RMSE) = ', num2str(err_nbit_uni(Nbit))]);
    subplot(2,4,Nbit+4); imshow(uint8(img_nbit_nuni(:,:,Nbit))); title([num2str(Nbit),'-bit Non-uniform Quantizaion Image']); 
    xlabel(['Quantization ERR (RMSE) = ', num2str(err_nbit_nuni(Nbit))]);
end
subplot(2,4,4); imshow(uint8(img)); title('Original Image'); % 원본 이미지 출력
subplot(2,4,8); % 1-3 bit Quantization Error (RMSE) 변화 그래프 출력
plot(1:3, err_nbit_uni(1:3), '-o');  hold on;
plot(1:3, err_nbit_nuni(1:3), '-or'); hold off;
title('Quantization Error (RSME) of Image'); xlabel('Quantized Bitrate'); ylabel('Quantization Error (RMSE)');
legend('Uniform', 'Non-uniform'); xlim([0 4]);

%%
% Bitrate에 따른 Quantization Error (RMSE) 비교 그래프 출력
figure(60); set(60, 'name', ['Uniform Quantization Error (RMSE) 1-',num2str(N),'bit']);
plot(1:N, err_nbit_uni, '-o'); hold on;
plot(1:N, err_nbit_nuni, '-or'); hold off;
title('Quantization Error (RSME) of Image'); xlabel('Quantized Bitrate'); ylabel('Quantization Error (RMSE)');
legend('Uniform', 'Non-uniform');

% 1~3 bits뿐만아니라 수식을 일반화하여 7bit까지 확인할 수 있도록 작성하였습니다. 
% 전체 이미지의 Quantization Error로 Root Mean Square Error(RSME)를 사용하였습니다.
% 픽셀 값이 편향적으로 분포한 이미지의 경우에 Non-uniform Quantizaion Error가 상대적으로 작아지는 것을 확인했습니다. 