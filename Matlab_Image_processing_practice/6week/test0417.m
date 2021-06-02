clear; close all; clc;
%%
img = imread('fig1.tif');
img = double(img);

img2 = [img(:,1), img , img(:, end)];
img2 = [img2(1,:);img2;img2(end,:)];


[H,W]= size(img(1:end-1,1:end-1));
lpf = [1 2 1; 2 4 2; 1 2 1]/16;

img_lpf = zeros(H,W);

for i =  1:H
    for j = 1:W
        img_lpf(i,j) = sum(sum(img2(i:i+2,j:j+2).*lpf));
    end
end
N=4;
img_samp = img_lpf(1:N:end,1:N:end);
%%

img_recon = myInterp(img_samp, N);

%figure(2); imshow(uint8(img_recon));

%%
hist = zeros(256);
for i = 1:H
    for j = 1:W
        hist(uint8(img_recon(i,j))+1) =  hist(uint8(img_recon(i,j))+1) + 1;
    end
end

cdf =  zeros(256);
for i = 1:256
    cdf(i) = sum(hist(1:i));
end
cdf = cdf/sum(hist);

%figure(1); plot(hist);figure(2); plot(cdf);

%% equalization
x1 = 90; x2 =135;
y1 = 0; y2 = 255;


a = (y2-y1)/(x2-x1);
b = -a*x1 + y1;

X = 0:255;
Y = a*X+b;
%Y = Y/255;

img_eq  = Y(uint8(img_recon)+1);

%figure(3); imshow(uint8(img_eq));

%% quantization

img_qt = zeros(H, W); 
img_eqqt = zeros(H, W);


% 128
% 64  192
% 
% 64 128 192
% 32 96  160 224
N = 2;

img_t = img(1:end-1, 1:end-1);

for k = 1:2^N
    img_qt((256/2^N)*(k-1)<= img_t & img_t < (256/2^N)*k) = (256/2^N)*(2*k-1)/2;
    img_eqqt((256/2^N)*(k-1)<= img_eq & img_eq < (256/2^N)*k) = (256/2^N)*(2*k-1)/2;
end

%% inverse
x1 = 0; x2 =255;
y1 = 90; y2 = 135;


a = (y2-y1)/(x2-x1);
b = -a*x1 + y1;

X = 0:255;
invY = a*X+b;

img_result  = invY(uint8(img_eqqt)+1);

figure(3); imshow(uint8(img_result));

%%

qt_error = log(sum(sum((img_qt-img_t).^2))/length(img(:)));
eqqt_error = log(sum(sum((img_result-img_t).^2))/length(img(:)));

disp(qt_error);disp(eqqt_error);




