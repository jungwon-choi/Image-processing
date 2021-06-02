close all; clear; clc;

img = imread('fig1.tif');
%img = rgb2gray(img);
img = double(img);

[H, W] = size(img);

N = 2; % 1~N bit 까지 

%%
hist = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist(img(i,j)+1) = hist(img(i,j)+1)+1;
    end
end
hist = hist/sum(hist);


cdf = zeros(256 , 1);
for i = 1:256
    cdf(i) = sum(hist(1:i));
end
cdf = cdf/cdf(end);

%%
% matlab histeq 함수
% img_eq = histeq(uint8(img),256);
% hist_eq = imhist(img_eq)./numel(img_eq);
% cdf_eq = cumsum(hist_eq);

% 직선 적용
X = 0:255;
Y = transF(X)/255;
img_myeq1 = uint8((255.0*Y(uint8(img)+1)));
hist_myeq1 = imhist(img_myeq1)./numel(img_myeq1);
cdf_myeq1 = cumsum(hist_myeq1);

% cdf함수 적용
img_myeq = uint8((255.0*cdf(uint8(img)+1)));
hist_myeq = imhist(img_myeq)./numel(img_myeq);
cdf_myeq = cumsum(hist_myeq);
% Transformation 함수 : cumulative sum of normalized histogram values

figure(80);
subplot(3,3,1); imshow(uint8(img));
subplot(3,3,2); plot(hist);xlim([1 256]);
subplot(3,3,3); plot(cdf);  hold on; plot(X+1,Y); hold off; xlim([1 256]); 
% subplot(3,3,4); imshow(uint8(img_eq));
% subplot(3,3,5); plot(hist_eq);xlim([1 256]);
% subplot(3,3,6); plot(cdf_eq);xlim([1 256]);
subplot(3,3,4); imshow(uint8(img_myeq1));
subplot(3,3,5); plot(hist_myeq1);xlim([1 256]);
subplot(3,3,6); plot(cdf_myeq);xlim([1 256]);
subplot(3,3,7); imshow(uint8(img_myeq));
subplot(3,3,8); plot(hist_myeq);xlim([1 256]);
subplot(3,3,9); plot(cdf_myeq);xlim([1 256]);
%%




%% Uniform Quantization

img_uniform_quantization = zeros(H,W,N);       % 각 Bitrate로 Uniform Quantization된 이미지를 저장할 공간
X = 1 : 255;                                                    % Quantization Function의 X축
Y_uni = zeros(length(X),N);                             % 각 Bitrate에 따른 Quantization Function [Y = Q(X)]
err_nbit_uni = zeros(1,N);                              % 각 Bitrate에 따른 Quantization Error (RMSE)

for Nbit = 1:N                         % 1~N Bitrate로 Quantizaion 
    img_temp = zeros(H,W);     % 연산을 위한 임시 이미지 공간 생성
    for k = 1 : 2^Nbit                % Quantizaion Step
        img_temp(round((256/2^Nbit)*(k-1)) <= img & img < round((256/2^Nbit)*k)) = round((256/2^Nbit)*(2*k-1)/2);
        % 조건식으로 Encoder Mapping 한 뒤 양 boundary의 중간 값을 대입하여 Decoder Mapping
        % 최대 픽셀값인 255를 양자화 범위에 포함하기 위하여 조건의 범위를 img < 256으로 설정
        Y_uni(round((256/2^Nbit)*(k-1)) <= X & X < round((256/2^Nbit)*k), Nbit) = round((256/2^Nbit)*(2*k-1)/2);
        % 해당 Bitrate일 때의 X에 대한 Quantization Function 생성
    end
    img_uniform_quantization(:,:,Nbit) = img_temp;  % 해당 Bitrate로 Uniform Quantization를 적용한 이미지 결과 저장
    err_nbit_uni(1,Nbit) = sqrt(sum(sum((img-img_uniform_quantization(:,:,Nbit)).^2))/length(img(:))); % Quantization Error (RMSE) 산출
end


err = sqrt(sum(sum((uint8(img)-img_myeq).^2))/length(img(:)));