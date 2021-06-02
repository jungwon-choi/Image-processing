function [Y_extend] = InterpLancoz(Y, N)

a = 3;                                                   % [-a, a] 지점 관여
x = zeros(2*a,3);                                   % 함수 값을 for문으로 입력하기 위한 배열 설정
len = length(Y);                                     % 정의역 index 최대 값
Y_extend = zeros(1,len*N);                    % 확장된 Y함수

for p = 1:N:len*N
    x(a,:) = Y(p);                                     % 현재 좌표 픽셀값

    for ll = a-1 : -1 : 1 
        if p-(a-ll)*N < 0                             % 정의역 범위를 벗어난 경우
            x(ll,:) = x(ll+1);                          % 다음 좌표의 값과 동일하게 처리 
        else                                              % 정의역 범위 내의 경우
            x(ll,:) = output_(p-(a-ll)*N);       % 해당 좌표값 선택
        end
    end

    for ll = a+1 : 2*a
        if p-(a-ll)*N > W*N                         % 정의역 범위를 벗어난 경우
            x(ll,:) = x(ll-1,:);                         % 이전 좌표의 값과 동일하게 처리 
        else                                               % 정의역 범위 내의 경우
            x(ll,:) = output_(p-(a-ll)*N);        % 해당 좌표값 선택
        end
    end

    % x1과 x2 사이의 픽셀에 대하여 Lanczos Interpolation 수식 적용
    for k = 1:N-1
        sum = 0;
        for ll = 1:2*a
            sum = sum + beta_func_lcz(k+(a-ll)*N, N, a)*x(ll,:);    % 각 픽셀값의 지분 합 
        end
        Y_extend(p+k) = sum;
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