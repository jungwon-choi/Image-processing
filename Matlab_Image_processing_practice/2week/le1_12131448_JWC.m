%% 
% 1/N (N = 4) 사이즈로 Sampling 된 이미지를
% 0th ~ 3rd order interpolation 기법을 각각 사용하여 Reconstruction 하기
%
% ⓒ 2018 최중원. All Right Reserved.
%% 
% Step1 : 이미지를 불러와서 Gaussian LPF를 적용하여 1/N 크기로 샘플링 하기
% Step1-1 : 이미지 불러오기
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

% Imn=Imn+1;figure(Imn); set(Imn, 'name', 'LPF없이 Simple 다운샘플링'); imshow(uint8(img_sampled))
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

%Imn=Imn+1;figure(Imn); set(Imn, 'name', 'LPF를 적용한 이미지'); imshow(uint8(img_lpf))
% Imn=Imn+1;figure(Imn); set(Imn, 'name', 'LPF를 적용한 다운샘플링');imshow(uint8(img_lpf_sampled))
%%
% Step2 : 0th ~ 3rd order Interpolation과 Cubic Convolution을 각각 적용하여 Reconstruction 수행
% Step2-1 : 가로 방향 적용

% 원래 가로 길이를 가진 빈 이미지 생성
img_recon_0th_col = zeros(H/N,W,D); 
img_recon_1st_col = zeros(H/N,W,D); % (이미지 처리 결과를 비교하기 위하여 1st 추가)
img_recon_2nd_col = zeros(H/N,W,D);
img_recon_3rd_col = zeros(H/N,W,D);
img_recon_cubconv_col = zeros(H/N,W,D);

% 다운샘플링된 데이터를 N간격으로 가로로 입력
img_recon_0th_col(:, 1:N:W, :) = img_lpf_sampled;
img_recon_1st_col(:, 1:N:W, :) = img_lpf_sampled;
img_recon_2nd_col(:, 1:N:W, :) = img_lpf_sampled;
img_recon_3rd_col(:, 1:N:W, :) = img_lpf_sampled;
img_recon_cubconv_col(:, 1:N:W, :) = img_lpf_sampled;

% 좌측 테두리 복원을 위하여 바로 근방의 값을 복사하여 비어있는 첫번째 열 값으로 사용 
img_recon_0th_col(:, 1, :) = img_lpf_sampled(:, 1+N, :);
img_recon_1st_col(:, 1, :) = img_lpf_sampled(:, 1+N, :);
img_recon_2nd_col(:, 1, :) = img_lpf_sampled(:, 1+N, :);
img_recon_3rd_col(:, 1, :) = img_lpf_sampled(:, 1+N, :);
img_recon_cubconv_col(:, 1, :) = img_lpf_sampled(:, 1+N, :);

% Imn=Imn+1;figure(Imn); set(Imn, 'name', 'Interpolation & Cubic Convolution 가로 적용전');imshow(uint8(img_recon_0th_col))

for i = 1:H/N % 1/N 배된 이미지 세로 픽셀 범위
    for j = 1:N:W % 원래 이미지의 가로 픽셀 범위
        
        % 0th order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 2개 지점의 픽셀 값이 필요
        
        x1_0th = img_recon_0th_col(i,j,:); % 현재 좌표 픽셀값

        % x2 선택
        if j+N > W % 이미지의 범위를 벗어났을 경우
             x2_0th = x1_0th; % 현재 좌표 픽셀 값을 다음 좌표 픽셀값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x2_0th = img_recon_0th_col(i,j+N,:); % 다음 좌표에 있는 픽셀 값을 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 0th order Interpolation 수식 적용
        for k = 1:N-1 
            if k < N/2 % x1과 x2 평균 값 미만의 경우 x1으로 적용
                img_recon_0th_col(i,j+k,:) = x1_0th;
            else % x1과 x2 평균 값 이상인 경우 x2로 적용 
                img_recon_0th_col(i,j+k,:) = x2_0th; 
            end
        end
        
        
        % 1st order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 2개 지점의 픽셀 값이 필요
        
        x1_1st = img_recon_1st_col(i,j,:); % 현재 좌표 픽셀값
        
        % x2 선택
        if j+N > W % 이미지의 범위를 벗어났을 경우
             x2_1st = x1_1st; % 현재 좌표 픽셀 값을 다음 좌표 픽셀값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x2_1st = img_recon_1st_col(i,j+N,:); % 다음 좌표에 있는 픽셀 값을 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 1th order Interpolation 수식 적용
        for k = 1:N-1 
             img_recon_1st_col(i,j+k,:) = (1-k/4)*x1_1st+(k/4)*x2_1st;
        end
      
        % 2nd order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if j-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_2nd = img_recon_2nd_col(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_2nd = img_recon_2nd_col(i,j-N,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_2nd = img_recon_2nd_col(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if j+N > W % 이미지의 범위를 벗어났을 경우
             x3_2nd = x2_2nd; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_2nd = img_recon_2nd_col(i,j+N,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if j+2*N > W % 이미지의 범위를 벗어났을 경우
             x4_2nd = x3_2nd; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_2nd = img_recon_2nd_col(i,j+2*N,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 2nd order Interpolation 수식 적용
        for k = 1:N-1 
            if k < N/2 % x1과 x2의 평균 값 미만의 경우 
                img_recon_2nd_col(i,j+k,:) = (0.5*(k/N - 0.5)^2).*x1_2nd ...  % x1에 대한 식 
                                                           + (3/4 - k/N^2).*x2_2nd ...       % x2에 대한 식
                                                           + (0.5*( k/N + 0.5)^2).*x3_2nd; % x3에 대한 식
                % RGB 3색을 각각 따로 계산하기 위하여 .*과 .^ 연산자 사용
                % x2를 중심으로 x1과 x3에 대한 식을 계산하기 위하여 각각 x+1, x-1로 평행이동    
            else % x1과 x2의 평균 값이상인 경우 
                img_recon_2nd_col(i,j+k,:) = (0.5*(k/N - 1.5)^2).*x2_2nd ...    % x2에 대한 식 
                                                           + (3/4 - (k/N - 1)^2).*x3_2nd ... % x3에 대한 식
                                                           + (0.5*(k/N - 0.5)^2).*x4_2nd;   % x4에 대한 식
                % x2를 중심으로 x3과 x4에 대한 식을 계산하기 위하여 각각 x-1, x-2로 평행이동
            end
        end
        
        
        % 3rd order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if j-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_3rd = img_recon_3rd_col(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_3rd = img_recon_3rd_col(i,j-N,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_3rd = img_recon_3rd_col(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if j+N > W % 이미지의 범위를 벗어났을 경우
             x3_3rd = x2_3rd; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_3rd = img_recon_3rd_col(i,j+N,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if j+2*N > W % 이미지의 범위를 벗어났을 경우
             x4_3rd = x3_3rd; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_3rd = img_recon_3rd_col(i,j+2*N,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 3rd order Interpolation 수식 적용
        for k = 1:N-1 
            img_recon_3rd_col(i,j+k,:) = ((1/6)*(1 - k/N)^3).*x1_3rd ...                                 % x1에 대한 식 
                                                       + (2/3 + 0.5*(k/N)^3 - (k/N)^2).*x2_3rd ...              % x2에 대한 식
                                                       + (2/3 - 0.5*(k/N-1)^3 - (k/N-1)^2).*x3_3rd ...       % x3에 대한 식
                                                       + ((1/6)*(k/N)^3).*x4_3rd;                                     % x4에 대한 식
            % RGB 3색을 각각 따로 계산하기 위하여 .*과 .^ 연산자 사용
            % x2를 중심으로 x1, x3, x4에 대한 식을 계산하기 위하여 각각 x+1, x-1, x-2로 평행이동
        end
        
        
        % Cubic Convolution 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if j-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_cubconv = img_recon_cubconv_col(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_cubconv = img_recon_cubconv_col(i,j-N,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_cubconv = img_recon_cubconv_col(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if j+N > W % 이미지의 범위를 벗어났을 경우
             x3_cubconv = x2_cubconv; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_cubconv = img_recon_cubconv_col(i,j+N,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if j+2*N > W % 이미지의 범위를 벗어났을 경우
             x4_cubconv = x3_cubconv; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_cubconv = img_recon_cubconv_col(i,j+2*N,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 Cubic Convolution 수식 적용
        
        a = -0.5; % 파라미터 
        
        for k = 1:N-1 
            img_recon_cubconv_col(i,j+k,:) = x1_cubconv.*(a*(k/N)^3 - 2*a*(k/N)^2 + a*(k/N)) ...                 % x1에 대한 식 
                                                           + x2_cubconv.*((a+2)*(k/N)^3 - (3+a)*(k/N)^2+1) ...                 % x2에 대한 식
                                                           + x3_cubconv.*(-1*(a+2)*(k/N)^3 + (2*a+3)*(k/N)^2-a*(k/N)) ...   % x3에 대한 식
                                                           + x4_cubconv.*(-a*(k/N)^3 + a*(k/N)^2);                                 % x4에 대한 식
            % RGB 3색을 각각 따로 계산하기 위하여 .* 연산자 사용
        end
        
    end
end

% Imn=Imn+1;figure(Imn); set(Imn, 'name', '0th 가로 적용후'); imshow(uint8(img_recon_0th_col))
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '1st 가로 적용후'); imshow(uint8(img_recon_1st_col))
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '2nd 가로 적용후'); imshow(uint8(img_recon_2nd_col))
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '3rd 가로 적용후'); imshow(uint8(img_recon_3rd_col))
% Imn=Imn+1;figure(Imn); set(Imn, 'name', 'cubconv 가로 적용후'); imshow(uint8(img_recon_cubconv_col))

%%
% Step2-2 : 세로 방향 적용

% 원래 사이즈를 가진 빈 이미지 생성
img_recon_0th_row = zeros(H,W,D);  
img_recon_1st_row = zeros(H,W,D);  
img_recon_2nd_row = zeros(H,W,D);  
img_recon_3rd_row = zeros(H,W,D);  
img_recon_cubconv_row = zeros(H,W,D);  

% 가로로 0th ~ 3rd order Interpolation을 적용한 데이터를 각각 N간격으로 세로로 입력
img_recon_0th_row(1:N:H, :, :) = img_recon_0th_col; 
img_recon_1st_row(1:N:H, :, :) = img_recon_1st_col; 
img_recon_2nd_row(1:N:H, :, :) = img_recon_2nd_col; 
img_recon_3rd_row(1:N:H, :, :) = img_recon_3rd_col;
img_recon_cubconv_row(1:N:H, :, :) = img_recon_3rd_col;

% 상단 테두리 복원을 위하여 바로 근방의 값을 복사하여 비어있는 첫번째 행 값으로 사용
img_recon_0th_row(1, :, :) = img_recon_0th_col(1+N, :, :); 
img_recon_1st_row(1, :, :) = img_recon_1st_col(1+N, :, :); 
img_recon_2nd_row(1, :, :) = img_recon_2nd_col(1+N, :, :); 
img_recon_3rd_row(1, :, :) = img_recon_3rd_col(1+N, :, :); 
img_recon_cubconv_row(1, :, :) = img_recon_3rd_col(1+N, :, :); 
 
%Imn=Imn+1;figure(Imn); set(Imn, 'name', '0th 세로 적용전');imshow(uint8(img_recon_0th_row))
%Imn=Imn+1;figure(Imn); set(Imn, 'name', '1st 세로 적용전');imshow(uint8(img_recon_1st_row))
%Imn=Imn+1;figure(Imn); set(Imn, 'name', '2nd 세로 적용전');imshow(uint8(img_recon_2nd_row))
%Imn=Imn+1;figure(Imn); set(Imn, 'name', '3rd 세로 적용전');imshow(uint8(img_recon_3rd_row))
%Imn=Imn+1;figure(Imn); set(Imn, 'name', 'cubconv 세로 적용전');imshow(uint8(img_recon_cubconv_row))

for i = 1:N:H % 원래 이미지의 세로 픽셀 범위
    for j = 1:W % 원래 이미지의 가로 픽셀 범위
         
        % 0th order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 2개 지점의 픽셀 값이 필요
        
        x1_0th = img_recon_0th_row(i,j,:); % 현재 좌표 픽셀값

        % x2 선택
        if i+N > H % 이미지의 범위를 벗어났을 경우
             x2_0th = x1_0th; % 현재 좌표 픽셀 값을 다음 좌표 픽셀값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x2_0th = img_recon_0th_row(i+N,j,:); % 다음 좌표에 있는 픽셀 값을 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 0th order Interpolation 수식 적용
        for k = 1:N-1 
            if k < N/2 % x1과 x2 평균 값 미만의 경우 x1으로 적용
                img_recon_0th_row(i+k,j,:) = x1_0th;
            else % x1과 x2 평균 값 이상인 경우 x2로 적용 
                img_recon_0th_row(i+k,j,:) = x2_0th; 
            end
        end
        
        
        % 1st order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 2개 지점의 픽셀 값이 필요
        
        x1_1st = img_recon_1st_row(i,j,:); % 현재 좌표 픽셀값
        
        % x2 선택
        if i+N > H % 이미지의 범위를 벗어났을 경우
             x2_1st = x1_1st; % 현재 좌표 픽셀 값을 다음 좌표 픽셀값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x2_1st = img_recon_1st_row(i+N,j,:); % 다음 좌표에 있는 픽셀 값을 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 1th order Interpolation 수식 적용
        for k = 1:N-1 
             img_recon_1st_row(i+k,j,:) = (1-k/N)*x1_1st+(k/N)*x2_1st;
        end
      
        % 2nd order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if i-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_2nd = img_recon_2nd_row(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_2nd = img_recon_2nd_row(i-N,j,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_2nd = img_recon_2nd_row(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if i+N > H % 이미지의 범위를 벗어났을 경우
             x3_2nd = x2_2nd; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_2nd = img_recon_2nd_row(i+N,j,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if i+2*N > H % 이미지의 범위를 벗어났을 경우
             x4_2nd = x3_2nd; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_2nd = img_recon_2nd_row(i+2*N,j,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 2nd order Interpolation 수식 적용
        for k = 1:N-1 
            if k < N/2 % x1과 x2의 평균 값 미만의 경우 
                img_recon_2nd_row(i+k,j,:) = (0.5*(k/N - 0.5)^2).*x1_2nd ...  % x1에 대한 식 
                                                           + (3/4 - k/N^2).*x2_2nd ...       % x2에 대한 식
                                                           + (0.5*( k/N + 0.5)^2).*x3_2nd; % x3에 대한 식
                % RGB 3색을 각각 따로 계산하기 위하여 .*과 .^ 연산자 사용
                % x2를 중심으로 x1과 x3에 대한 식을 계산하기 위하여 각각 x+1, x-1로 평행이동    
            else % x1과 x2의 평균 값이상인 경우 
                img_recon_2nd_row(i+k,j,:) = (0.5*(k/N - 1.5)^2).*x2_2nd ...    % x2에 대한 식 
                                                           + (3/4 - (k/N - 1)^2).*x3_2nd ... % x3에 대한 식
                                                           + (0.5*(k/N - 0.5)^2).*x4_2nd;   % x4에 대한 식
                % x2를 중심으로 x3과 x4에 대한 식을 계산하기 위하여 각각 x-1, x-2로 평행이동
            end
        end
        
        
        % 3rd order Interpolation 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if i-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_3rd = img_recon_3rd_row(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_3rd = img_recon_3rd_row(i-N,j,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_3rd = img_recon_3rd_row(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if i+N > H % 이미지의 범위를 벗어났을 경우
             x3_3rd = x2_3rd; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_3rd = img_recon_3rd_row(i+N,j,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if i+2*N > H % 이미지의 범위를 벗어났을 경우
             x4_3rd = x3_3rd; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_3rd = img_recon_3rd_row(i+2*N,j,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 3rd order Interpolation 수식 적용
        for k = 1:N-1 
            img_recon_3rd_row(i+k,j,:) = ((1/6).*(1 - k/N).^3).*x1_3rd ...                               % x1에 대한 식 
                                                       + (2/3 + 0.5.*(k/N).^3 - (k/N).^2).*x2_3rd ...              % x2에 대한 식
                                                       + (2/3 - 0.5.*(k/N-1).^3 - (k/N-1).^2).*x3_3rd ...       % x3에 대한 식
                                                       + ((1/6).*(k/N).^3).*x4_3rd;                                     % x4에 대한 식
            % RGB 3색을 각각 따로 계산하기 위하여 .*과 .^ 연산자 사용
            % x2를 중심으로 x1, x3, x4에 대한 식을 계산하기 위하여 각각 x+1, x-1, x-2로 평행이동
        end
        
        
        % Cubic Convolution 적용 알고리즘 ----------------------------------------
        % 수식 적용을 위해 4개 지점의 픽셀 값이 필요
        
        % x1 선택
        if i-N < 0 % 이미지의 범위를 벗어났을 경우
            x1_cubconv = img_recon_cubconv_row(i,j,:); % 현재 좌표의 픽셀값과 동일하게 처리
        else
            x1_cubconv = img_recon_cubconv_row(i-N,j,:); % 이전 좌표의 픽셀값 선택
        end
        
        x2_cubconv = img_recon_cubconv_row(i,j,:); % 현재 좌표 픽셀값
        
        % x3 선택
        if i+N > H % 이미지의 범위를 벗어났을 경우
             x3_cubconv = x2_cubconv; % 현재 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x3_cubconv = img_recon_cubconv_row(i+N,j,:); % 다음 좌표에 있는 픽셀 값 선택
        end
        
        % x4 선택
        if i+2*N > H % 이미지의 범위를 벗어났을 경우
             x4_cubconv = x3_cubconv; % 현재 기준으로 다음 좌표 픽셀 값과 동일하게 처리
        else % 이미지의 범위 내의 경우
             x4_cubconv = img_recon_cubconv_row(i+2*N,j,:); % 다다음 좌표에 있는 픽셀 값 선택
        end
        
        % x1과 x2 사이의 픽셀에 대하여 Cubic Convolution 수식 적용
        
        a = -0.5; % 파라미터 
        
        for k = 1:N-1
            img_recon_cubconv_row(i+k,j,:) = x1_cubconv.*(a*(k/N)^3 - 2*a*(k/N)^2+a*(k/N)) ...               % x1에 대한 식 
                                                           + x2_cubconv.*((a+2)*(k/N)^3 - (a+3)*(k/N)^2+1) ...                 % x2에 대한 식
                                                           + x3_cubconv.*(-1*(a+2)*(k/N)^3 + (2*a+3)*(k/N)^2-a*(k/N)) ...   % x3에 대한 식
                                                           + x4_cubconv.*(-a*(k/N)^3 + a*(k/N)^2);                                 % x4에 대한 식
                                                            % RGB 3색을 각각 따로 계산하기 위하여 .* 연산자 사용
        end
        
    end
end


% Imn=Imn+1;figure(Imn); set(Imn, 'name', '0th order Interpolation Reconstruction'); 
% set(Imn,'units','normalized','outerposition',[0.15 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로) 
% imshow(uint8(img_recon_0th_row), 'border','tight')
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '1st order Interpolation Reconstruction');
% set(Imn,'units','normalized','outerposition',[0.3 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로)
% imshow(uint8(img_recon_1st_row), 'border','tight')
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '2nd order Interpolation Reconstruction');
% set(Imn,'units','normalized','outerposition',[0.45 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로)
% imshow(uint8(img_recon_2nd_row), 'border','tight')
% Imn=Imn+1;figure(Imn); set(Imn, 'name', '3rd order Interpolation Reconstruction');
% set(Imn,'units','normalized','outerposition',[0.6 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로)
% imshow(uint8(img_recon_3rd_row), 'border','tight')
% Imn=Imn+1;figure(Imn); set(Imn, 'name', 'Cubic Convolution Reconstruction'); 
% set(Imn,'units','normalized','outerposition',[0.75 1 (W-10)/scrWH(3) H/scrWH(4)]); % 창 겹침 방지를 위해 위치 조정(%로)
% imshow(uint8(img_recon_cubconv_row), 'border','tight')

Imn=Imn+1;figure(Imn); set(Imn, 'name', '이미지 전체 비교');
set(Imn,'units','normalized','outerposition',[0 0 1 1]); 
subplot(2,3,1); imshow(uint8(img), 'border', 'tight'); title('Original Image');
subplot(2,3,2); imshow(uint8(img_recon_0th_row), 'border','tight'); title('0th order Interpolation'); 
subplot(2,3,3); imshow(uint8(img_recon_1st_row), 'border','tight'); title('1st order Interpolation');
subplot(2,3,4); imshow(uint8(img_recon_2nd_row), 'border','tight'); title('2nd order Interpolation');
subplot(2,3,5); imshow(uint8(img_recon_3rd_row), 'border','tight'); title('3rd order Interpolation'); 
subplot(2,3,6); imshow(uint8(img_recon_cubconv_row), 'border','tight'); title('Cubic Convolution'); 