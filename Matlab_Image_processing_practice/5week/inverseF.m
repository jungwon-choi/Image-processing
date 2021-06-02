function [invY] = inverseF(Y)
% 입력 도메인이 0~255(N = 256)이고 출력 도메인이 0~1인 함수의 역함수를 구하는 함수 

N = length(Y);

invY = zeros(1,N);
for ll = 1 : N
    invY(uint8((N-1)*Y(ll))+1) = ll;  
    % y=x축을 대칭으로 X와 Y를 서로 교환
end

invY = EmptyInterp(invY);
% 1:1 대응이 아닌 지점의 경우 Interpolation 수행

invY = invY/(N-1); 
% 다시 출력 도메인을 0~1로 만들기 위해 Normalization

end