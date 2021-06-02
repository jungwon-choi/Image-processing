%% 그동안 한거 복습
clear; close all; clc
%% 이미지 불러오기
img = imread('fig1.tif');
img = double(img);

[H,W] = size(img);
%% 가우시안 필터 적용
img_gaussian = zeros(H,W);

g = [1 2 1;
       2 4 2;
       1 2 1]/16;
   
img2 = [img(:,1), img, img(:, end)];
img2 = [img2(1,:); img2; img2(end,:)];


for i = 1:H
    for j = 1:W
        img_gaussian(i,j)=sum(sum(img2(i:i+2,j:j+2).*g));
    end
end

%% 샘플링 
img_sampled = img_gaussian(1:2:end, 1:2:end);

figure;
subplot(1,2,1); imshow(uint8(img2));
subplot(1,2,2); imshow(uint8(img_sampled));
%% bilnear interpolation

img_recon = zeros(ceil(H/2),W);
img_recon(:,1:2:end) = img_sampled;

for i = 1:H/2
    for j = 1:2:W-1
        if j+2 > W
            img_recon(i,j+1) = img_recon(i,j);
        else
            img_recon(i,j+1) = (img_recon(i,j)+img_recon(i,j+2))/2;
        end
    end
end

img_recon2 = zeros(H,W);
img_recon2(1:2:end,:) = img_recon;

for i = 1:2:H-1
    for j = 1:W
        if i+2 > H
            img_recon2(i+1,j) = img_recon2(i,j);
        else
            img_recon2(i+1,j) = (img_recon2(i,j)+img_recon2(i+2,j))/2;
        end
    end
end

figure;
subplot(1,2,1); imshow(uint8(img_recon));
subplot(1,2,2); imshow(uint8(img_recon2));
%% Histogram

img_recon2 = round(img_recon2); % hist(index) 정수 만드려고
hist = zeros(1,256);
% hist = zeros(256,1);
for i = 1 : H
    for j = 1:W
        hist(img_recon2(i,j) + 1) = hist(img_recon2(i,j) + 1) + 1;
    end
end
plot(hist);

%% cdf

cdf = zeros(1,256);
for i = 1:256
    cdf(i) = sum(hist(1:i));
end

cdf = cdf/cdf(end);
plot(cdf);

%% cdf 근사직선 함수 만들기

x1 = 90; y1 = 0;
x2 = 135; y2 = 255;

a = (y2-y1)/(x2-x1);
b= -a*x1+y1;

Y = min(y2, max(y1,a.*(0:255)+b));
plot(Y); xlim([0 255]); ylim([0 255]);
%% Image Equalization
% img_equ = 255*min(y2, max(y1,a.*img_recon2+b));
img_equ = Y(img_recon2+1);
imshow(uint8(img_equ));

%% Quantization

img_1b = img;
img_1b(img_1b<128) = 64;
img_1b(img_1b>=128) = 192;

img_equ_1b = img_equ;
img_equ_1b(img_equ_1b<128) = 64;
img_equ_1b(img_equ_1b>=128) = 192;


img_2b = img;
img_2b(img_2b<64) = 32;
img_2b(img_2b>=64 & img_2b<128) = 96;
img_2b(img_2b>=128 & img_2b<196) = 162;
img_2b(img_2b>=196) = 226;

img_equ_2b = img_equ;
img_equ_2b(img_equ_2b<64) = 32;
img_equ_2b(img_equ_2b>=64 & img_2b<128) = 96;
img_equ_2b(img_equ_2b>=128 & img_2b<196) = 162;
img_equ_2b(img_equ_2b>=196) = 226;

figure;
subplot(2,2,1); imshow(uint8(img_1b));
subplot(2,2,2); imshow(uint8(img_2b));
subplot(2,2,3); imshow(uint8(img_equ_1b));
subplot(2,2,4); imshow(uint8(img_equ_2b));
% 0일 경우 이전 값으로 업데이트 
% 새로운 값으로 업데이트

%% inverse transformation function

x1 = 0; y1 = 90;
x2 = 255; y2 = 135;

a = (y2-y1)/(x2-x1);
b= -a*x1+y1;

invY = min(y2, max(y1,a.*(0:255)+b));
plot(invY); xlim([0 255]); ylim([0 255]);
%% inverse transformation
img_result_1b = invY(img_equ_1b+1);
img_result_2b = invY(img_equ_2b+1);
figure;
subplot(2,2,1); imshow(uint8(img_1b));
subplot(2,2,2); imshow(uint8(img_2b));
subplot(2,2,3); imshow(uint8(img_result_1b));
subplot(2,2,4); imshow(uint8(img_result_2b));