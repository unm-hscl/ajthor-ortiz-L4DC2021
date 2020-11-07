function dx = nonlinear_CWH(x, u)

n = 0.00113;
m = 300;

Ts = 20;

% ddx - 3*n^2*x - 2*n*dy + Fx/m;
% ddy + 2*n*dx + Fy/m;
%
% x1 = x;
% x2 = y;
% x3 = dx;
% x4 = dy;
%
% dx1 = x3;
% dx2 = x4;
% dx3 = 3*n^2*x1 + 2*n*x4 - Fx/m;
% dx4 = -2*n*x3 - Fy/m;
%
% u1 = Fx;
% u2 = Fy;

dx = zeros(4, 1);
dx(1) = x(3)                            + Ts*x(3) + 1E-4*randn(1);
dx(2) = x(4)                            + Ts*x(4) + 1E-4*randn(1);
dx(3) = 3*n^2*x(1) + 2*n*x(4) - u(1)/m  + Ts/x(3) + 5E-8*randn(1);
dx(4) = -2*n*x(3) - u(2)/m              + Ts/x(4) + 5E-8*randn(1);

end
