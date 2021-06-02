function [Y_stuffed] = EmptyInterp(Y)
% 함수 값이 0으로 empty한 경우 이를 보간 해주는 함수

len = length(Y);                                % 정의역 index 최대 값

start_flag = 0;                                 % 보간 시작지점 체크 조건
interval = 1;                                     % start와 end 지점 간격
end_flag = 0;                                   % 보간 끝지점 체크 조건

for pos = 1:len
    
    % start_pos와 end_pos 지점 사이의 empty개수 카운트
    if start_flag == 1 && Y(pos) == 0
        interval = interval + 1; 
    end
    
    % 스타트 지점 생성 후 empty가 아닌 다음 값 나오면 Interpolation 시작 조건 Set
    if  end_flag == 0 && Y(pos) ~= 0
        end_pos = pos;
        end_flag = 1;
    end
    
    % empty가 아닌 지점일 경우 스타트 지점으로 설정
    if start_flag == 0 && Y(pos) ~= 0
        start_pos = pos;
        start_flag = 1;
    end
    
    % 연속으로 empty가 아닌 경우
    if start_flag == 1 && end_flag == 1 && interval ==1
        start_pos = pos;
        end_flag = 0;
    end
    
    % 정의역 끝에 도달하여 Interpolation을 수행 못할 경우 
    if pos + 1 > len && interval > 1
        for k = 1:interval-1
            Y(start_pos+k) = Y(start_pos); % 가장 최근 empty 아닌 값으로 채움
        end
    end
    
    % Lancoz Interpolation 시작
    if start_flag ==1 && end_flag ==1 
            
        %       x1       x2        x3       x4
        %                start      end

        % x2 
        x2 = Y(start_pos);                 % start_pos 지점 함수 값
        x1 = x2;                                % start_pos 지점 함수 값과 동일하게 처리 
        x3 = Y(end_pos);                   % end_pos 지점 함수 값 선택
        x4 = x3;                                % end_pos 지점 함수 값 동일하게 처리

        % start_pos와 end_pos 사이의 지점의 함수 값을 Cubic Convolution Interpolation 으로 보간
        for k = 1:interval-1
            Y(start_pos+k) = beta_func_ccv(k+interval, interval)*x1...
                                    +beta_func_ccv(k, interval)*x2...
                                    +beta_func_ccv(k-interval, interval)*x3...
                                    +beta_func_ccv(k-2*interval, interval)*x4;
        end
    
        start_pos = end_pos;
        start_flag = 1;
        interval = 1;
        end_flag = 0;
    end
    
Y_stuffed = Y;
end

% Cubic Convoltion Interpolation beta 함수
function [b] = beta_func_ccv(k,n)
    alpha = -0.5;
    k = k/n;
    if (0 <= abs(k)) && (abs(k) < 1)
        b = (alpha+2)*abs(k)^3 - (alpha+3)*abs(k)^2 + 1;
    elseif (1 <= abs(k)) && (abs(k) < 2)
        b = alpha*abs(k)^3 - 5*alpha*abs(k)^2 + 8*alpha*abs(k) - 4*alpha;
    else
        b = 0;
    end
end

end