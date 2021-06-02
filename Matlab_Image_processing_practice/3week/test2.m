function [output] = interp(img, n, method) 
%----------------------------------------------------------------------
% interp : 축소된 이미지를 보간법을 활용하여 Reconstruction 을 수행하는 함수
%
% img : 입력받은 이미지
% n : 복원 크기 배수
% method : 복원을 수행할 방법 
%----------------------------------------------------------------------
    
% 입력받은 옵션에 따라 이미지 복원 방법 선택
if strcmp(method, '0th')
    key = 0;
elseif strcmp(method, '1st')
    key = 1;
elseif strcmp(method, '2nd')
    key = 2;
elseif strcmp(method, '3rd')
    key = 3;
elseif strcmp(method, 'CubicConv')
    key = 4;
elseif strcmp(method, 'Lanczos')
    key = 5;
else % 입력받은 복원 옵션 올바르지 않을 경우
    output = 'NULL';
    disp("아래의 Reconstruction 방법을 선택해주세요.")
    disp("method : '0th', '1st', '2nd', '3rd', 'CubicConv', 'Lanczos' ")
    return
end

img = double(img);                          % 이미지 픽셀 값을 실수로 변환 (보간 수식 적용을 위함)
output = recon(img, n);                    % 가로(열)에 대하여 복원 수행
output = permute(output,[2 1 3]);   % 이미지의 행열을 서로 대칭시켜 변환
output = recon(output, n);               % 세로(행)에 대하여 복원 수행
output = permute(output,[2 1 3]);   % 이미지의 행열을 다시 서로 대칭시켜 원래 이미지로 변환

function [output_] = recon(img_, n)     % 이미지를 열에 대하여 N배로 복원을 수행
    [H, W, D] = size(uint8(img_));          % 이미지의 크기를 추출하여 저장
    output_ = zeros(H, W*n, D);             % 복원될 이미지 공간을 0으로 초기화
    output_(:, 1:n:W*n,:) = img_;            % 기존 이미지를 n간격으로 입력
    output_(:, 1, :) = img_(:,1+n,:);         % 좌측의 검은선을 방지하기 위해 초기값 근처 픽셀과 동일하게 복사

    for i = 1:H             % 모든 행에 대하여 수행
        for j = 1:n:W*n  % 해당 행에 대하여 n 간격으로 열을 이동

            switch key % method에 따라 복원 알고리즘을 선택하여 수행

                case 0  % 0th Interpolation method
                    
                    %       x1       x2
                    %   현재픽셀 
                    
                    % x1
                    x1 = output_(i,j,:);              % 현재 위치의 픽셀 값
                    
                    % x2 선택
                    if j+n >W                            % 이미지 범위를 초과한 경우 
                            x2 = x1;                     % 바로 이전 픽셀 값을 대입
                    else                                   % 이미지의 범위 내의 경우
                            x2 = output_(i,j+n,:);  % 다음 위치의 픽셀 값
                    end

                    for k = 1 : n-1                    % x1 ~ x2 사이의 픽셀에 대하여 보간법 수행
                         output_(i,j+k,:) = beta_func0(k, n)*x1...
                                                +beta_func0(k-n, n)*x2;
                    end

                case 1 % 1st Interpolation method
                    
                    %       x1       x2
                    %   현재픽셀 
                    
                    % x1
                    x1 = output_(i,j,:);              % 현재 위치의 픽셀 값
                    
                    % x2 선택
                    if j+n >W                            % 이미지 범위를 초과한 경우 
                            x2 = x1;                     % 바로 이전 픽셀 값을 대입
                    else                                   % 이미지의 범위 내의 경우
                            x2 = output_(i,j+n,:);  % 다음 위치의 픽셀 값
                    end

                    for k = 1 : n-1                    % x1 ~ x2 사이의 픽셀에 대하여 보간법 수행
                         output_(i,j+k,:) = beta_func1(k, n)*x1...
                                                 +beta_func1(k-n, n)*x2;
                    end

                case 2 % 2nd Interpolation method
                    
                    %       x1       x2        x3       x4
                    %              현재픽셀
                    
                    % x2 
                    x2 = output_(i,j,:);             % 현재 좌표 픽셀값
                    
                    % x1 선택
                    if j-n < 0                          % 이미지의 범위를 벗어난 경우
                        x1 = x2;        % 현재 좌표의 픽셀 값과 동일하게 처리 
                    else                                 % 이미지의 범위 내의 경우
                        x1 = output_(i,j-n,:);    % 이전 좌표의 픽셀값 선택
                    end
                    
                    % x3 선택
                    if j+n > W                         % 이미지의 범위를 벗어난 경우
                         x3 = x2;                      % 현재 좌표 픽셀 값과 동일하게 처리
                    else                                 % 이미지의 범위 내의 경우
                         x3 = output_(i,j+n,:);   % 다음 좌표에 있는 픽셀 값 선택
                    end

                    % x4 선택
                    if j+2*n > W                      % 이미지의 범위를 벗어난 경우
                         x4 = x3;                       % 다음 좌표 픽셀 값과 동일하게 처리
                    else                                  % 이미지의 범위 내의 경우
                         x4 = output_(i,j+2*n,:); % 다다음 좌표에 있는 픽셀 값 선택
                    end

                    % x1과 x2 사이의 픽셀에 대하여 2nd order Interpolation 수식 적용
                    for k = 1:n-1
                        output_(i,j+k,:) = beta_func2(k+n, n)*x1...
                                                +beta_func2(k, n)*x2...
                                                +beta_func2(k-n, n)*x3...
                                                +beta_func2(k-2*n, n)*x4;
                    end

                case 3 % 3rd Interpolation method
                    
                    %       x1       x2        x3       x4
                    %              현재픽셀
                    
                    % x2 
                    x2 = output_(i,j,:);             % 현재 좌표 픽셀값
                    
                    % x1 선택
                    if j-n < 0                          % 이미지의 범위를 벗어난 경우
                        x1 = x2;        % 현재 좌표의 픽셀 값과 동일하게 처리 
                    else                                 % 이미지의 범위 내의 경우
                        x1 = output_(i,j-n,:);    % 이전 좌표의 픽셀값 선택
                    end
                    
                    % x3 선택
                    if j+n > W                         % 이미지의 범위를 벗어난 경우
                         x3 = x2;                      % 현재 좌표 픽셀 값과 동일하게 처리
                    else                                 % 이미지의 범위 내의 경우
                         x3 = output_(i,j+n,:);   % 다음 좌표에 있는 픽셀 값 선택
                    end

                    % x4 선택
                    if j+2*n > W                      % 이미지의 범위를 벗어난 경우
                         x4 = x3;                       % 다음 좌표 픽셀 값과 동일하게 처리
                    else                                  % 이미지의 범위 내의 경우
                         x4 = output_(i,j+2*n,:); % 다다음 좌표에 있는 픽셀 값 선택
                    end

                    % x1과 x2 사이의 픽셀에 대하여 2nd order Interpolation 수식 적용
                    for k = 1:n-1
                        output_(i,j+k,:) = beta_func3(k+n, n)*x1...
                                                +beta_func3(k, n)*x2...
                                                +beta_func3(k-n, n)*x3...
                                                +beta_func3(k-2*n, n)*x4;
                    end

                case 4 % Cubic Convolution Interpolation method
                    
                    %       x1       x2        x3       x4
                    %              현재픽셀
                    
                    % x2 
                    x2 = output_(i,j,:);             % 현재 좌표 픽셀값
                    
                    % x1 선택
                    if j-n < 0                          % 이미지의 범위를 벗어난 경우
                        x1 = x2;        % 현재 좌표의 픽셀 값과 동일하게 처리 
                    else                                 % 이미지의 범위 내의 경우
                        x1 = output_(i,j-n,:);    % 이전 좌표의 픽셀값 선택
                    end
                    
                    % x3 선택
                    if j+n > W                         % 이미지의 범위를 벗어난 경우
                         x3 = x2;                      % 현재 좌표 픽셀 값과 동일하게 처리
                    else                                 % 이미지의 범위 내의 경우
                         x3 = output_(i,j+n,:);   % 다음 좌표에 있는 픽셀 값 선택
                    end

                    % x4 선택
                    if j+2*n > W                      % 이미지의 범위를 벗어난 경우
                         x4 = x3;                       % 다음 좌표 픽셀 값과 동일하게 처리
                    else                                  % 이미지의 범위 내의 경우
                         x4 = output_(i,j+2*n,:); % 다다음 좌표에 있는 픽셀 값 선택
                    end

                    % x1과 x2 사이의 픽셀에 대하여 2nd order Interpolation 수식 적용
                    for k = 1:n-1
                        output_(i,j+k,:) = beta_func_ccv(k+n, n)*x1...
                                                +beta_func_ccv(k, n)*x2...
                                                +beta_func_ccv(k-n, n)*x3...
                                                +beta_func_ccv(k-2*n, n)*x4;
                    end

                case 5 % Lanczos Interpolation method
                    a = 3; 
                    x = zeros(2*a,3);
                    
                    x(a,:) = output_(i,j,:);
                    
                    for ll = a-1 : 1 
                        if j-(a-ll)*n < 0                          % 이미지의 범위를 벗어난 경우
                            x(ll,:) = x(ll+1);        % 현재 좌표의 픽셀 값과 동일하게 처리 
                        else                                 % 이미지의 범위 내의 경우
                            x(ll,:) = output_(i,j-(a-ll)*n,:);    % 이전 좌표의 픽셀값 선택
                        end
                    end
                    
                    for ll = a+1 : 2*a
                        if j+ll*n > W                          % 이미지의 범위를 벗어난 경우
                            x(ll,:) = x(ll-1,:);        % 현재 좌표의 픽셀 값과 동일하게 처리 
                        else                                 % 이미지의 범위 내의 경우
                            x(ll,:) = output_(i,j-(a-ll)*n,:);    % 이전 좌표의 픽셀값 선택
                        end
                    end

                    for k = 1:n-1
                        sum = 0;
                        for ll = 1:2*a
                            sum = sum + beta_func_lcz(k-(a-ll)*n, n, a)*x(ll,:);
                        end
                        output_(i,j+k,:) = sum;
                    end

                otherwise
                    warning('이 경우는 없을듯?!')
                    output_ = img_;
                    return
            end
        end
    end
end

% 0th-order Interpolation beta 함수
function [b] = beta_func0(k,n)
    k = k/n;
    if (-1/2 < k) && (k <= 1/2)
        b = 1;
    else
        b = 0;
    end
end

% 1st-order Interpolation beta 함수
function [b] = beta_func1(k,n)
    k = k/n;
    if(-1 < k) && (k < 0)
        b = k+1;
    elseif (0 <= k) &&(k <= 1)
        b = 1-k;
    else
        b = 0;
    end
end

% 2nd-order Interpolation beta 함수
function [b] = beta_func2(k,n)
    k = k/n;
    if (-3/2 < k) && (k <= -1/2)
        b = 0.5*(k + 1.5)^2;
    elseif (-1/2 < k) && (k <= 1/2)
        b = 3/4 - k^2;
    elseif (1/2 < k) && (k <= 3/2)
        b = 0.5*(k - 1.5)^2;
    else
        b = 0;
    end
end

% 3rd-order Interpolation beta 함수
function [b] = beta_func3(k,n)
    k = k/n;
    if (0 <= abs(k)) && (abs(k) < 1)
        b = 2/3+0.5*abs(k)^3 - k^2;
    elseif (1 <= abs(k)) && (abs(k) < 2)
        b = 1/6*(2 - abs(k))^3;
    else
        b = 0;
    end
end

% Cubic Convoltion Interpolation beta 함수
function [b] = beta_func_ccv(k,n)
    a = -0.5;
    k = k/n;
    if (0 <= abs(k)) && (abs(k) < 1)
        b = (a+2)*abs(k)^3 - (a+3)*abs(k)^2 + 1;
    elseif (1 <= abs(k)) && (abs(k) < 2)
        b = a*abs(k)^3 - 5*a*abs(k)^2 + 8*a*abs(k) - 4*a;
    else
        b = 0;
    end
end

% Lanczos Interpolation beta 함수
function [b] = beta_func_lcz(k,n, a)
    k = k/n;
    if abs(k) < a
        b = sinc(k)*sinc(k/a);
    else
        b = 0;
    end
end

end