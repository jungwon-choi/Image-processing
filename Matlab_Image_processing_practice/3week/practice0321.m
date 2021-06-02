close all; clear; clc;

img = imread('fig0.png'); % fig0.png 이미지 파일 불러와서 img에 저장하기
img = double(img); % img의 데이터를 실수형으로 변환 (실수단위 연산을 위함)
[H, W, D] = size(img); % 불러온 이미지 크기 추출 (Height, Width, Dimension)
scrWH = get(0, 'screensize'); % 창 위치 조절을 위한 스크린 사이즈 추출


N = 4; % downsizing 배수

img_sampled = img(1:N:end, 1:N:end,:); 
% Low Pass Filter 적용하지 않고 샘플링한 경우 -> 픽셀 사이의 값의 차이가 큰 경우 이미지가 거칠어 보임
% Simple downsampling 이라고도 함

Imn = 0; % Image number : 창 구분을 위함
% Imn=Imn+1; figure(Imn);  set(Imn, 'name', '원본 이미지'); % 이미지 창 띄우기
% set(Imn,'units','normalized','outerposition',[0 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로)
% imshow(uint8(img), 'border', 'tight') % 이미지 크기에 맞추어 출력
% figure 창에 이미지를 unsigned 정수형으로 변환하여 8bit로 출력 (imshow 8bit 정수형 사용)
%Imn=Imn+1;figure(Imn); set(Imn, 'name', 'LPF없이 Simple 다운샘플링'); imshow(uint8(img_sampled))
%%
% Step1-2 : Gaussian LPF 필터를 적용하여 Down 샘플링하기

% lpf = ones(3,3,3)/9;  % Uniform Weight Moving average LPF 필터
lpf = [1,2,1; ...
         2,4,2; ...
         1,2,1]/16; % Gaussian LPF 필터

img_lpf = zeros(H,W,D); % LPF 필터를 적용할 이미지 공간 생성

for i = 1:H-2   
    for j = 1:W-2
        img_lpf(i+1,j+1,:) = sum(sum( img(i:i+2,j:j+2,:).*lpf, 1),2); 
        % .곱을 통하여 3x3필터를 적용하여 배열의 합을 현재 픽셀 값으로 저장
        % 테두리 픽셀의 경우 필터를 적용할 외부의 값이 없기 때문에 테두리 안쪽 범위 공간만 필터 적용
    end
end

img_lpf_sampled = img_lpf(1:N:end, 1:N:end,:); % Low Pass Filter를 적용하여 샘플링

Imn=Imn+1; figure(Imn);  set(Imn, 'name', 'sample'); % 이미지 창 띄우기
imshow(uint8(img_lpf_sampled), 'border', 'tight') % 이미지 크기에 맞추어 출력
%%
% img_recon_0th = interp_0th_(img_lpf_sampled, N);
img_recon_1st = interp_1st_(img_lpf_sampled, N);
% img_recon_2nd = interp_2nd_(img_lpf_sampled, N);
% img_recon_3rd = interp_3rd_(img_lpf_sampled, N);
% img_recon_ccv = cubic_conv_(img_lpf_sampled, N);
 
Imn=Imn+1;figure(Imn); set(Imn, 'name', '이미지 전체 비교');
% set(Imn,'units','normalized','outerposition',[0 0 1 1]); 
subplot(2,3,1); imshow(uint8(img), 'border', 'tight'); title('Original Image');
% subplot(2,3,2); imshow(uint8(img_recon_0th), 'border','tight'); title('0th order Interpolation'); 
subplot(2,3,3); imshow(uint8(img_recon_1st), 'border','tight'); title('1st order Interpolation');
% subplot(2,3,4); imshow(uint8(img_recon_2nd), 'border','tight'); title('2nd order Interpolation');
% subplot(2,3,5); imshow(uint8(img_recon_3rd), 'border','tight'); title('3rd order Interpolation'); 
% subplot(2,3,6); imshow(uint8(img_recon_ccv), 'border','tight'); title('Cubic Convolution'); 