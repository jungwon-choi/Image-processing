close all;

img = imread('fig0.png');  % 이미지 불러오기
[H,W,D] = size(img);

img_x4 = img(1:4:end, 1:4:end, :);

% img_recon = zeros(H,W,D);
% img_recon(1:4:H,1:4:W,:);


% img_recon = zeros(H/4,W,D);
% 
% img_recon(:,1:4:W,:) = img_x4; % 새로 배치해줌
% 
% figure(8);
% imshow(uint8(img_recon))
% 
% for i = 1:H/4
% %     for j = 1:4:W-4
%         
%     for j = 1:4:W
%         
%         x1 = img_recon(i,j,:);
% %         x2 = img_recon(i,j+4,:);
%         if j+4 > W
%             x2 = x1;
%         else
%             x2 = img_recon(i,j+4,:);
%         end
%         
%         
%         
%         for k = 1:3
%             img_recon(i,j+k,:) = (1-k/4)*x1 + (k/4)*x2;
%         end
%     end
% end
% 
% img_recon2 = zeros(H,W,D);
% 
% img_recon2(1:4:H,:,:) = img_recon;
% figure(9);
% imshow(uint8(img_recon2))
% 
% for i = 1:4:H
%     for j = 1:W
%         x1 = img_recon2(i,j,:);
%         if i+4 > H
%             x2 = x1;
%         else
%             x2 = img_recon2(i+4,j,:);
%         end
%         
%         for k = 1:3
%             img_recon2(i+k,j,:) = (1-k/4)*x1+(k/4)*x2;
%         end
%     end
% end


%%
% figure;
% imshow(img, 'border', 'tight')

lpf =  ones(3,3,3)/9;
lpf_gaussian =  [1,2,1;2,4,2;1,2,1]/16;


img = double(img);  % 보통 이미지 읽자마자 더블로 바꾸는게 일반적 uint8로 사용할 일이 거의없음

img_lpf = zeros(H,W,D);
img_gaussian = zeros(H,W,D);

for i = 1:H-2       %  테두리 제외
    for j = 1:W-2
        img_lpf(i+1,j+1,:) = sum(sum(img(i:i+2, j:j+2,:).*lpf,1),2);
        img_gaussian(i+1,j+1,:) = sum(sum(img(i:i+2, j:j+2,:).*lpf_gaussian,1),2);
%         img_lpf(i+1,j+1,:) = sum(sum(img(i:i+2, j:j+2,:).*lpf));
    end
end

% imfilter(img, ) %% 필터 내장함수

% figure; imshow(uint8(img_lpf), 'border', 'tight') % imshow를 하려면 무조건 unit8 형식
% figure; imshow(uint8(img_gaussian), 'border', 'tight')

img_input1 = zeros(H,W,D);
img_input2 = zeros(H,W,D);

img_input1 = img_lpf(1:4:end, 1:4:end, :);
img_input2 = img_gaussian(1:4:end, 1:4:end, :);

out1 = interp_0th_(img_lpf, 4);
out2 = interp_0th_(img_gaussian, 4);

figure; imshow(out1)
figure; imshow(out2)

%%
% 이미지 작은 사이즈로 샘플링
% for i = 1:5
% 
%     img_sampled = imresize(img(1:2^i:end, 1:2^i:end, :), 2^i, `bilinear`);
% %% `bilinear`  `bicubic`
%     
%     figure(100)
%     imshow(img_sampled, 'border', 'tight')
%     pause;
% end