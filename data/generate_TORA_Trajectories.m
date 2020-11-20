% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

Ts = 0.1;           % Sample Time
N = 3;              % Time horizon
Duration = 20;      % Simulation horizon

stoc = false;        % Should the trajectories be stochastic?

radius = 0.1;

fnameNN = 'TORA_NN.nt';

global simulation_result;

X = double.empty(4, 0);
Y = double.empty(4, 0);

end_states = double.empty(4, 0);

for m=1:50

    x0 = 0.6 + radius*rand(1);
    y0 = -0.6 - radius*rand(1);
    z0 = -0.3 - radius*rand(1);
    w0 = 0.5 + radius*rand(1);

    x = [x0;y0;z0;w0;];

    simulation_result = x;

    options = optimoptions('fmincon','Algorithm','sqp','Display','none');
    uopt = zeros(N,1);

    u_max = 0;

    % Apply the control input constraints
    LB = -3*ones(N,1);
    UB = 3*ones(N,1);

    x_now = zeros(4,1);
    x_next = zeros(4,1);

    x_now = x;

    for ct = 1:(Duration/Ts)

        u = TORA_NN_output(x_now, 10, 1, fnameNN);

        Y = [Y, x_now];

        x_next = TORA_Dynamics(x_now, Ts, u);

        if stoc
            % x_next = x_next + 0.0025*randn(4, 1);
            % x_next = x_next + 0.0025*exprnd(1.5, 4, 1);
            x_next = x_next + 0.01*betarnd(2, 0.5, 4, 1);
        end

        X = [X, x_next];

        x = x_next;
        x_now = x_next;

    end

end

if stoc
    save('TORA_stoc.mat', 'X', 'Y');
else
    save('TORA_det.mat', 'X', 'Y');
end

% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

function [y] =  TORA_NN_output(x,offset,scale_factor,name)

file = fopen(name,'r');
file_data = fscanf(file,'%f');
no_of_inputs = file_data(1);
no_of_outputs = file_data(2);
no_of_hidden_layers = file_data(3);
network_structure = zeros(no_of_hidden_layers+1,1);
pointer = 4;
for i = 1:no_of_hidden_layers
    network_structure(i) = file_data(pointer);
    pointer = pointer + 1;
end
network_structure(no_of_hidden_layers+1) = no_of_outputs;


weight_matrix = zeros(network_structure(1), no_of_inputs);
bias_matrix = zeros(network_structure(1),1);

% READING THE INPUT WEIGHT MATRIX
for i = 1:network_structure(1)
    for j = 1:no_of_inputs
        weight_matrix(i,j) = file_data(pointer);
        pointer = pointer + 1;
    end
    bias_matrix(i) = file_data(pointer);
    pointer = pointer + 1;
end

% Doing the input transformation
g = zeros(no_of_inputs,1);
g = x;
g = weight_matrix * g;
g = g + bias_matrix(:);
g = do_thresholding(g);


for i = 1:(no_of_hidden_layers)

    weight_matrix = zeros(network_structure(i+1), network_structure(i));
    bias_matrix = zeros(network_structure(i+1),1);

    % READING THE WEIGHT MATRIX
    for j = 1:network_structure(i+1)
        for k = 1:network_structure(i)
            weight_matrix(j,k) = file_data(pointer);
            pointer = pointer + 1;
        end
        bias_matrix(j) = file_data(pointer);
        pointer = pointer + 1;
    end

    % Doing the transformation
    g = weight_matrix * g;
    g = g + bias_matrix(:);
    g = do_thresholding(g);

end

y = g-offset;
y = y * scale_factor;
fclose(file);

end

% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

function[h] = do_thresholding(r)
    [size_1, size_2]  = size(r);
    out = zeros(size_1,1);
    for i = 1:size_1
       if(r(i) > 0)
         out(i) = r(i);
       else
          out(i) = 0;
       end
    end
    h = out;
end
