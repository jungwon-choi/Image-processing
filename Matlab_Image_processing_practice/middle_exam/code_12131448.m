%% 이미지 불러오기 및 초기 설정
img = imread('nir_img.png');
img = img(1:end-1, :); % 3배수 맞게 이미지 자르기A
img = double(img);
[H, W] = size(img);

N = 3; % 3배수
%% Q1. 1/3 sub-sampling with gaussian LPF

% 이미지 외각 copy padding
img2 = [img(:,1), img, img(:,end)];
img2 = [img2(1,:); img2; img2(end,:)];

% Gaussian 필터
lpf = [1 2 1; 2 4 2; 1 2 1]/16;

img_lpf = zeros(H,W);

% Gaussian 필터 Convolution 적용
for i = 1:H
    for j = 1:W
        img_lpf(i,j) = sum(sum(img2(i:i+2,j:j+2).*lpf));
    end
end

% 이미지 1/3 크기로 sub-sampling
img_samp = img_lpf(1:N:end, 1:N:end);

%% Q2-1. Histogram equalizaition

% 히스토그램 제작
hist = zeros(1, 256);
for i = 1:H/N
    for j = 1:W/N
        hist(uint8(img_samp(i,j))+1) = hist(uint8(img_samp(i,j))+1) +1;
    end
end

% CDF 함수 제작
cdf = zeros(1,256);
for i = 1:256
    cdf(i) = sum(hist(1:i));
end
cdf = cdf/cdf(end);

% Transfer 함수 제작 (cdf 근사)
x1 = 184; x2 = 250;
y1 = 0; y2 = 255;

a = (y2-y1)/(x2-x1);
b = -a*x1+y1;
X = 0:255;

Y = max(0, min(255, a.*X+b));

% 이미지에 Transfer 함수 적용
img_eq = Y(uint8(img_samp)+1);

%% 제대로 평활화 되었는지 테스트
hist_eq = zeros(1, 256);
for i = 1:H/N
    for j = 1:W/N
        hist_eq(uint8(img_eq(i,j))+1) = hist_eq(uint8(img_eq(i,j))+1) +1;
    end
end


cdf_eq = zeros(1,256);
for i = 1:256
    cdf_eq(i) = sum(hist_eq(1:i));
end
cdf_eq = cdf_eq/cdf_eq(end);

figure(1); plot(hist_eq); figure(2); plot(cdf_eq);
%% Q2-2. 3-bit quantization

img_qt = zeros(H/N, W/N);

Nbit = 3;

% 양자화 과정을 일반화 시킨 반복문
for k = 1:2^Nbit
    img_qt((256/2^Nbit)*(k-1) <= img_eq & img_eq < (256/2^Nbit)*k) = (256/2^Nbit)*(2*k-1)/2;
    % 양자화 단계의 중간 값을 취하여 Error 최소화
end
    
%% Q2-3. Inverse Transfer

% Inverse Transfer 함수 제작 (Inverse cdf 근사)
x1 = 0; x2 = 255;
y1 = 184; y2 = 250;

a = (y2-y1)/(x2-x1);
b = -a*x1+y1;
X = 0:255;

invY = max(0, min(255, a.*X+b));

% 이미지에 Inverse Transfer 함수 적용
img_ivt = invY(uint8(img_qt)+1);
figure(5);plot(X, Y/255, X, invY/255); xlim([0,255]); ylim([0,1]);
%% 히스토그램 재검사
% hist_eq2 = zeros(1, 256);
% for i = 1:H/N
%     for j = 1:W/N
%         hist_eq2(uint8(img_ivt(i,j))+1) = hist_eq2(uint8(img_ivt(i,j))+1) +1;
%     end
% end
% 
% 
% cdf_eq2 = zeros(1,256);
% for i = 1:256
%     cdf_eq2(i) = sum(hist_eq(1:i));
% end
% cdf_eq2 = cdf_eq2/cdf_eq2(end);
% 
% figure(3); plot(hist_eq2); figure(4); plot(cdf_eq2);
%% Q3. up-sampling with cubic convolution interpolation

% 가로 방향 Interpolation 
img_recon = zeros(H/N, W);
img_recon(:, 1:N:W) = img_ivt; 

for i = 1: H/N
    for j = 1:N:W
        
        % 두번째 좌표 픽셀값
        x2 = img_recon(i,j);     
        
        % 이미지 범위 내에서 첫번째 좌표 픽셀값 선택
        if j - N < 0
            x1 = x2;
        else
            x1 = img_recon(i,j - N);
        end
        
        % 이미지 범위 내에서 세번째 좌표 픽셀값 선택
        if j + N > W
            x3 = x2;
        else
            x3 = img_recon(i,j + N);
        end
        
        % 이미지 범위 내에서 네번째 좌표 픽셀값 선택
        if j + 2*N > W
            x4 = x3;
        else
            x4 = img_recon(i,j + 2*N);
        end
        
        % x1과 x2 사이의 픽셀 값을 Cubic Convolution을 이용하여 Interpolation 수행
        a = -0.5;
        for k = 1 : N-1            
            t = k/N;
            img_recon(i,j+k) = (a*t^3 - 2*a*t^2 + a*t)*x1 ...
                                     + ((a+2)*t^3 - (3+a)*t^2 + 1)*x2 ...
                                     + (-(a+2)*t^3 + (2*a+3)*t^2 - a*t)*x3 ...
                                     + (-a*t^3 + a*t^2)*x4;
        end      
    end
end

% 세로 방향 Interpolation 
img_recon2 = zeros(H, W);
img_recon2(1:N:H,:) = img_recon; 

for i = 1:N:H
    for j = 1:W
        
        % 두번째 좌표 픽셀값
        x2 = img_recon2(i,j);
        
        % 이미지 범위 내에서 첫번째 좌표 픽셀값 선택
        if i - N < 0
            x1 = x2;
        else
            x1 = img_recon2(i - N,j);
        end
        
        % 이미지 범위 내에서 세번째 좌표 픽셀값 선택
        if i + N > H
            x3 = x2;
        else
            x3 = img_recon2(i + N,j);
        end
        
        % 이미지 범위 내에서 네번째 좌표 픽셀값 선택
        if i + 2*N > H
            x4 = x3;
        else
            x4 = img_recon2(i + 2*N,j);
        end
        
        % x1과 x2 사이의 픽셀 값을 Cubic Convolution을 이용하여  Interpolation 수행
        a = -0.5;
        for k = 1 : N-1
            t = k/N;
            img_recon2(i+k,j) = (a*t^3 - 2*a*t^2 + a*t)*x1 ...
                                     + ((a+2)*t^3 - (3+a)*t^2 + 1)*x2 ...
                                     + (-(a+2)*t^3 + (2*a+3)*t^2 - a*t)*x3 ...
                                     + (-a*t^3 + a*t^2)*x4;
        end
    end
end

%% Q4. Quantization Error (PSNR)

% Mean Square Error
MSE = sum(sum(img-img_recon2).^2)/length(img(:));
% PSNR Error
qt_error = 10*log10(255^2/MSE);
disp(['Quantization Error (MSE) : ', num2str(MSE)]);
disp(['Quantization Error (PSNR) : ', num2str(qt_error)]);

%% Result Plot
figure(100);
subplot(2,4,1); imshow(uint8(img)); title('Original Image');
subplot(2,4,2); imshow(uint8(img_samp)); title('Sub-Sampling Image with Gaussian LPF');
subplot(2,4,3); plot(hist); title('Image Histogram');
subplot(2,4,4); plot(X,Y/255, X, cdf); title('Transfer Function');
subplot(2,4,5); imshow(uint8(img_eq)); title('Historam Equalized Image');
subplot(2,4,6); imshow(uint8(img_qt)); title('3-bit Quantized Image');
subplot(2,4,7); imshow(uint8(img_ivt)); title('Inverse Transform Image');
subplot(2,4,8); imshow(uint8(img_recon2)); title('Reconstruction Image');
xlabel(['Quantization Error : ', num2str(MSE),' (MSE)   ', num2str(qt_error),' (PSNR)']);

%% Result Image Save
imwrite(uint8(img_recon2), 'output_12131448.bmp');
