        elseif key == 2
        
            a = 3;                                                           % [-a, a] 지점 관여
            x = zeros(2*a);                                           % 함수 값을 for문으로 입력하기 위한 배열 설정

            %   x(1)  ...  x(a-1)    x(a)    x(a+1)  ...  x(2a)
            %                          현재픽셀
            
            x(a) = Y(start_pos);                                     % 현재 좌표 픽셀값

            % 이전 좌표값 선택
            for ll = a-1 : -1 : 1 
                if start_pos-(a-ll) <= 0                               % 정의역 범위를 벗어난 경우
                    x(ll) = x(ll+1);                                      % 다음 좌표의 값과 동일하게 처리 
                else                                                          % 정의역 범위 내의 경우
                    x(ll) = Y(start_pos-(a-ll));                    % 해당 좌표값 선택
                end
            end

            % 이후 좌표값 선택
            for ll = a+1 : 2*a
                if pos-(a-ll) > len                                       % 정의역 범위를 벗어난 경우
                    x(ll) = x(ll-1);                                    % 이전 좌표의 값과 동일하게 처리 
                else                                                          % 정의역 범위 내의 경우
                    x(ll) = Y((end_pos-1)-(a-ll));                             % 해당 좌표값 선택
                end
            end
            disp(interval)
            % start_pos와 end_pos 사이의 지점의 함수 값을 Lanczos Interpolation으로 보간
            for k = 1:interval-1
                sum = 0;
                for ll = 1:2*a
                    sum = sum + beta_func_lcz(k+(a-ll)*interval, interval, a)*x(ll);    % 각 지점의 함수 값의 지분 합 
                end
                Y(start_pos+k) = sum;
            end
        
        else
           disp('Method Error');
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
