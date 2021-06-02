img = imread('fig1.tif');
%img = imread('fig0.png');
img = double(img);



lpf = [1 2 1; 2 4 2; 1 2 1]/16;


img2 = [img(:,1), img, img(:,end)];
img2 = [img2(1,:); img2; img2(end,:)];
img = img(1:end-1, 1:end-1);
[H, W] = size(img);
img_lpf = zeros(H,W);


for i = 1:H
    for j = 1:W
        img_lpf(i,j) = sum(sum(img2(i:i+2,j:j+2).*lpf));
    end
end


%figure(2); imshow(uint8(img_lpf));
%%
N = 4;

img_sampled = img(1:N:end, 1:N:end);

%figure(2); imshow(uint8(img_sampled));

%%


img_cubcov = pre_interCC(img_sampled,N);


%figure(2); imshow(uint8(img_cubcov));
%%

