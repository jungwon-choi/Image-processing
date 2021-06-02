function [Y] = LinearTransF(X)
% cdf를 근사시킨 직선 함수를 만드는 함수

% 두 점의 좌표
x1 = 91;
y1= 0;
x2 = 137; % 127이 더 근사
y2= 255;

% 직선의 기울기와 y절편
a = (y2-y1)/(x2-x1);
b = -a*x1+y1;

Y = max(y1, min(y2, a*X+b));

end