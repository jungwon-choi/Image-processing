%% 5week HW : Histogram Equalization & Uniform Quantization
%    Copyright 2018, Jungwon Choi, INHA Electronics
close all; clear; clc;

img = imread('fig1.tif');
%img = rgb2gray(img);
img = double(img);
[H, W] = size(img);

N = 5; % 1~N bit 까지 

%% 1. compute histogram %%
% 이미지의 픽셀값별 개수 카운트

hist = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist(img(i,j)+1) = hist(img(i,j)+1)+1;
        % 벡터의 시작 인덱스가 1부터 이기 때문에 +1 수행
    end
end
hist = hist/sum(hist); % Normalization 수행

% 과정1를 다음과 같이 대체 가능 >> hist = imhist(img)/numel(img);

%% 2. get transformaion function %%
% Transformation 함수 : cumulative sum of normalized histogram values
% (Cumulative Distribution Function 활용)

% cdf 함수 제작
cdf = zeros(256 , 1);
for i = 1:256
    cdf(i) = sum(hist(1:i));
    % 픽셀별 카운트 값을 누적해서 저장
end
cdf = cdf/cdf(end);

% 과정2를 다음과 같이 대체 가능 >> cdf = cumsum(hist); 

%% 3. histogram equalization %%

% cdf로 equalization 수행
img_eq = uint8((255.0*cdf(uint8(img)+1)));
% cdf 함수 출력이 0 ~ 1 이므로 픽셀값에 대응하도록 255 곱 수행

%% (Supplementary) 
% 비교를 위해 cdf 근사 직선 함수로 equalization 수행
X = 0:255;
Y_eqf =  LinearTransF(X); 
img_eql = uint8((Y_eqf(uint8(img)+1)));
% 다음과 같이 대체 가능 >> img_eql = max(y1,min(y2, ((y2-y1)/(x2-x1))*(img-x1)+y1));

%%%%%%%%%% 변환 결과 확인 %%%%%%%%%%
hist_eq = imhist(img_eq)./numel(img_eq);
cdf_eq = cumsum(hist_eq);
hist_eql = imhist(img_eql)./numel(img_eql);
cdf_eql = cumsum(hist_eql);

figure(100); set(100, 'name', 'Equalization result comparation','units','normalized','outerposition', [0 0 1 1]);
subplot(3,3,1); imshow(uint8(img)); title('Original Image');
subplot(3,3,2); plot(hist);xlim([1 256]);
subplot(3,3,3); plot(X, cdf, 'b', X, Y_eqf/255, 'r'); xlim([1 256]); 
subplot(3,3,4); imshow(uint8(img_eq)); title('historam equalization Image (cdf)');
subplot(3,3,5); plot(hist_eq);xlim([1 256]);
subplot(3,3,6); plot(cdf_eq);xlim([1 256]);
subplot(3,3,7); imshow(uint8(img_eql)); title('historam equalization Image (linear cdf function)');
subplot(3,3,8); plot(hist_eql);xlim([1 256]);
subplot(3,3,9); plot(cdf_eql);xlim([1 256]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. uniform quantization
% 히스토그램 평활화한 이미지와 원본 이미지를 각각 uniform quantization 수행

img_just_uniqnt = zeros(H,W,N);       % 각 bitrate로 uniform quantization된 이미지를 저장할 공간 (Original image)
img_hist_uniqnt = zeros(H,W,N);       % 각 bitrate로 uniform quantization된 이미지를 저장할 공간 (Histgram equalized image)

for Nbit = 1:N                                   % 1~N bitrate로 quantizaion 
    img_temp1 = zeros(H,W);             % 연산을 위한 임시 이미지 공간 생성 (Original image)
    img_temp2 = zeros(H,W);             % 연산을 위한 임시 이미지 공간 생성 (Histgram equalized image)
    for k = 1 : 2^Nbit                         % Quantizaion Step
        img_temp1(round((256/2^Nbit)*(k-1)) <= img & img < round((256/2^Nbit)*k)) = round((256/2^Nbit)*(2*k-1)/2);
        img_temp2(round((256/2^Nbit)*(k-1)) <= img_eq & img_eq < round((256/2^Nbit)*k)) = round((256/2^Nbit)*(2*k-1)/2);
        % 조건식으로 encoder mapping 한 뒤 양 boundary의 중간 값을 대입하여 decoder mapping
        % 최대 픽셀값인 255를 양자화 범위에 포함하기 위하여 조건의 범위를 img < 256으로 설정
    end
    img_just_uniqnt(:,:,Nbit) = img_temp1;  % 해당 bitrate로 uniform quantization를 적용한 이미지 결과 저장 (Original image)
    img_hist_uniqnt(:,:,Nbit) = img_temp2;  % 해당 bitrate로 uniform quantization를 적용한 이미지 결과 저장 (Histgram equalized image)
end


%% 5. inverse transform %%

invcdf = inverseF(cdf);
% inverse transform을 위한 cdf의 역함수를 구함

img_hist_uniqnt_inv = uint8((255.0*invcdf(uint8(img_hist_uniqnt)+1)));
% 각 이미지별로 inverse transform 수행

figure(150); figure(150); set(150, 'name', 'Inverse result');
plot(X, cdf , 'b-'); hold on; 
plot(X, invcdf, 'r-' ); hold off;
legend('Transformaion Function', 'Inverse Transformaion Function'); xlim([0,255]); ylim([0,1]);
%% 6. Calculate quantization error %%
err_just_uniqnt = zeros(1,N);    % 각 Bitrate에 따른 quantization error (RMSQE) (Original image)
err_hist_uniqnt = zeros(1,N);    % 각 Bitrate에 따른 quantization error (RMSQE) (Histgram equalized image)

for Nbit = 1:N
    % quantization error (RMSE) 산출
    err_just_uniqnt(Nbit) = sqrt(sum(sum((img-double(img_just_uniqnt(:,:,Nbit))).^2))/length(img(:))); 
    err_hist_uniqnt(Nbit) = sqrt(sum(sum((img-double(img_hist_uniqnt_inv(:,:,Nbit))).^2))/length(img(:))); 
end

%% 7. Display the result %%

figure(200); set(200, 'name', 'Histogram Equalization Quantization result comparation','units','normalized','outerposition', [0 0 1 1]);
subplot(3,N,1); imshow(uint8(img)); title('Original Image');
for ll = 1:N
    subplot(3,N, N+ll); imshow(uint8(img_just_uniqnt(:,:,ll))); title(['Unifrom ',num2str(ll),'bit quantization']); xlabel(['quantization error (RMSE) : ',num2str(err_just_uniqnt(ll))]);
    subplot(3,N, 2*N+ll); imshow(uint8(img_hist_uniqnt_inv(:,:,ll))); title(['Histogram Equalization ',num2str(ll),'bit quantization']); xlabel(['quantization error (RMSE) : ',num2str(err_hist_uniqnt(ll))]);
end
subplot(3,N,2); plot(1:N, err_just_uniqnt, 'b', 1:N, err_hist_uniqnt, 'r' ); % bitrate에 따른 Quantization error 변화 
title('Quantization Error (RSME) of Image'); xlabel('Quantized Bitrate'); ylabel('Quantization Error (RMSE)');
xlim([0 N+1]); ylim('auto'); legend('Uniform quantization', 'Histogram equalizaion quantizatioin');