clear; %close all;  % 이전에 띄운 이미지 창 닫기

% 원하는 이미지 불러오기
img = imread('fig0.png');   

N = 4; % 샘플링 간격

% 이미지를 1/4 크기로 샘플링 (자체 추가 제작 함수)
img_lpf_sampled = samp(img, N, 'Gaussian'); 


% 이미지를 본래 크기로 복원 
img_recon_0th = interp(img_lpf_sampled, N, '0th');             % 0th-order Interpolation 사용 
img_recon_1st = interp(img_lpf_sampled, N, '1st');              % 1st-order Interpolation 사용 
img_recon_2nd = interp(img_lpf_sampled, N, '2nd');             % 2nd-order Interpolation 사용 
img_recon_3rd = interp(img_lpf_sampled, N, '3rd');              % 3rd-order Interpolation 사용 
img_recon_ccv = interp(img_lpf_sampled, N, 'CubicConv');     % Cubic Convolution Interpolation 사용 
img_recon_lcz = interp(img_lpf_sampled, N, 'Lanczos');         % Lanczos Interpolation 사용

% 이미지 창 설정 (전체화면)
figure(99); set(99, 'name', '이미지 전체 비교','units','normalized','outerposition', [0 0 1 1]); 

% 각 이미지를 subplot으로 동시에 띄우기
subplot(2,4,1); imshow(uint8(img), 'border', 'tight'); title('Original Image');
subplot(2,4,5); imshow(uint8(img_recon_0th), 'border','tight'); title('0th order Interpolation'); 
subplot(2,4,6); imshow(uint8(img_recon_1st), 'border','tight'); title('1st order Interpolation');
subplot(2,4,7); imshow(uint8(img_recon_2nd), 'border','tight'); title('2nd order Interpolation');
subplot(2,4,8); imshow(uint8(img_recon_3rd), 'border','tight'); title('3rd order Interpolation'); 
subplot(2,4,2); imshow(uint8(img_recon_ccv), 'border','tight'); title('Cubic Convolution Interpolation'); 
subplot(2,4,3); imshow(uint8(img_recon_lcz), 'border','tight'); title('Lanczos Interpolation'); 