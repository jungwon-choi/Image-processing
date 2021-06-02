close all; clear; clc;

img = imread('fig1.tif');
%img = rgb2gray(img);
img = double(img);

[H, W] = size(img);

%%

hist = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist(img(i,j)+1) = hist(img(i,j)+1)+1;
    end
end


cdf = zeros(256 , 1);
for i = 1:256
    cdf(i) = sum(hist(1:i));
end
cdf = cdf/cdf(end);

%round(255./(1+exp( -([0:255]-130)/32)));
%%

% (91,0), (137,1)

X = 0:255;

f = cdf';

img_eq = round(img.*f(img+1));

Y_eq = round(255*f(X+1))/255;
Y_log = round(105.88*log10(X+1))/255;
Y_ivlog = round(20.1646*((1.01).^X-1))/255;
r = 10;
c = 255^(1-r);
Y_gamma = round(c*X.^r)/255;

hist_eq = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist_eq(img_eq(i,j)+1) = hist_eq(img_eq(i,j)+1)+1;
    end
end


cdf_eq = zeros(256 , 1);
for i = 1:256
    cdf_eq(i) = sum(hist_eq(1:i));
end
cdf_eq = cdf_eq/cdf_eq(end);

figure(80);
subplot(2,2,1); imshow(uint8(img_eq));
subplot(2,2,2); plot(X,cdf', '-b'); hold on; plot(X,Y_eq, '-r'); plot(X,Y_log, '-g'); plot(X,Y_ivlog, '-k'); plot(X,Y_gamma, '-y');  hold off; xlim([0 255])
subplot(2,2,3); plot(hist_eq);
subplot(2,2,4); plot(cdf_eq);
%%
img2 = round(255./(1+exp( -(img-130)/32)));
hist2 = zeros(256,1);


% 1/1+e(-(x-c)/b) 식만드려고 직선 3개말고
for i = 1:H
    for j = 1:W
        hist2(img2(i,j)+1) = hist2(img2(i,j)+1)+1;
    end
end
%hist2 = hist2 / sum(hist2);


cdf2 = zeros(256 , 1);
for i = 1:256
    cdf2(i) = sum(hist2(1:i));
end
cdf2 = cdf2/cdf2(end);


figure;
subplot(3,2,1); plot(hist);
subplot(3,2,2); plot(hist2);
subplot(3,2,3); plot(cdf);
subplot(3,2,4); plot(cdf2);
subplot(3,2,5); imshow(uint8(img));
subplot(3,2,6); imshow(uint8(img2));

%%
figure;

%직선3개만들기 MAX를 활용 max(5, [1 9 3 0 2 3])  5로 클램프
%max

for k = 1:255
    img3 = round(255./(1+exp( -(img-110)/k)));
    hist3 = zeros(256,1);
    for i = 1:H
        for j = 1:W
            hist3(img3(i,j)+1) = hist3(img3(i,j)+1)+1;
        end
    end
    
    cdf3 = zeros(256 , 1);
    for i = 1:256
        cdf3(i) = sum(hist3(1:i));
    end
    cdf3 = cdf3/cdf3(end);

    subplot(1,3,1); imshow(uint8(img3)); xlabel(['b =', num2str(k)]);
    subplot(1,3,2); plot(hist3);
    subplot(1,3,3); plot(cdf3);
    pause(0.1);
end

%%
X = 1:256;
%img4 = round(max(0,min(255,255*(2e-2*img-90/50))));
%Y = round(max(0,min(255,255.*(2e-2*X-90/50))));
img4 = round(255./(1+exp( -(img-130)/32)));
Y = round(1./(1+exp( -(X-130)/10)));
hist4 = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist4(img4(i,j)+1) = hist4(img4(i,j)+1)+1;
    end
end

cdf4 = zeros(256 , 1);
for i = 1:256
    cdf4(i) = sum(hist4(1:i));
end
cdf4 = cdf4/cdf4(end);

figure(50);
subplot(1,3,1); imshow(uint8(img4));
subplot(1,3,2); plot(hist4);
subplot(1,3,3); plot(cdf, '-b'); hold on; plot(X,Y, '-r');

%%





% b : 1 ~ 10
% c : 105 ~ 115
X = 0:255;

opt_c = -1;
opt_b = -1;
min_err = 10000;

for c = 105:0.005:115
    for b = 1:0.005:10
    
    Y = round(255./(1+exp( -(X-c)/b)))/255;
    err = sqrt(sum((Y'-cdf).^2)/256);
    if err < min_err
        min_err = err;
        opt_c = c;
        opt_b = b;
    end
    
    end
end

img5 = round(255./(1+exp( -(img-opt_c)/opt_b)));
Y = round(255./(1+exp( -(X-opt_c)/opt_b)))./255;

hist5 = zeros(256,1);
for i = 1:H
    for j = 1:W
        hist5(img5(i,j)+1) = hist5(img5(i,j)+1)+1;
    end
end

cdf5 = zeros(256 , 1);
for i = 1:256
    cdf5(i) = sum(hist5(1:i));
end
cdf5 = cdf5/cdf5(end);

figure(70);
subplot(2,2,1); imshow(uint8(img5));
subplot(2,2,2); plot(cdf, '-b'); hold on; plot(X+1,Y, '-r'); hold off;
xlabel(['c = ', num2str(opt_c), ' b = ', num2str(opt_b), ' err = ', num2str(min_err)]);
subplot(2,2,3); plot(hist5);
subplot(2,2,4); plot(cdf5);

%%


