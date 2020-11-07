% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

function final_val = invertedPendulum_dynamics(x0, T, u, stoc)

global simulation_result;

function dxdt = tora(t, x)
    e = 0.1;
    if stoc
        dxdt = [x(2); -x(1) + e * sin(x(3)); x(4); u] + 0.01*randn(4, 1);
    else
        dxdt = [x(2); -x(1) + e * sin(x(3)); x(4); u];
    end
end

[t ,y] = ode45(@tora, [0 T], x0) ;

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end
