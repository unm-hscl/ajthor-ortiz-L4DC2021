% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

function final_val = TORA_Dynamics(x0, T, u)

global simulation_result;

function dxdt = tora(t, x)
    e = 0.1;
    dxdt = [x(2); -x(1) + e * sin(x(3)); x(4); u];
end

[t ,y] = ode45(@tora, [0 T], x0) ;

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end
