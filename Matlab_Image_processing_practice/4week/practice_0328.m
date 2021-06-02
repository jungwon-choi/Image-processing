close all; clear;

img = imread('fig0.png');   
img = rgb2gray(img);
img = double(img);

[H, W] = size(img);
%%
% 양자화 실습

img_1b = zeros(H,W);

for i = 1:H
    for j = 1:W
        if img(i,j) < 128
            img_1b(i,j) = 64;   % 0~128 중간값
        else
            img_1b(i,j) = 64+128;   % 128~255 중간값
        end
    end
end

img_1b_mat = zeros(H,W);
img_1b_mat(img<128) = 64;
img_1b_mat(img>=128) = 192;


img_1b_1 = zeros(H,W);
img_1b_1(img<64) = 64;
img_1b_1(img>=64) = 192;

img_1b_2 = zeros(H,W);
img_1b_2(img<192) = 64;
img_1b_2(img>=192) = 192;

m = mean(img(:)); % n차원을 1차원으로 만들어줌

img_1b_mean = zeros(H,W);
img_1b_mean(img<m) = 64;
img_1b_mean(img>=m) = 192;

figure;
subplot(2,3,1); imshow(uint8(img))
% subplot(2,3,2); imshow(uint8(img_1b))
subplot(2,3,3); imshow(uint8(img_1b_mat))
% subplot(2,3,4); imshow(uint8(img_1b_1))
% subplot(2,3,5); imshow(uint8(img_1b_2))
subplot(2,3,6); imshow(uint8(img_1b_mean))

log(sqrt(sum(sum((img-img_1b).^2)))) % 이거 비트별로 그래프 그려보기

%%
% 2bits 양자화

img_t = zeros(H,W);
img_t(0<=img & img<64) = 32;
img_t(64<=img & img<128) = 96;
img_t(128<=img & img<196) = 162;
img_t(196<=img & img<255) = 226;
figure; imshow(uint8(img_t))
% img = floor(img/N)

%%
% Non-Uniform 양자화 (평균값을 이용)

th1 = mean(img(:));
th0 = mean(mean(img(img<th1)));
th2 = mean(mean(img(img>=th1)));

img2b = img;
img2b(img2b<th0) = th0/2;
img2b(th0<img2b & img2b<th1) = (th1+th0)/2;


%non_uniform 이 uniform 보다 에러가 작아야함