% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

no_of_traces = 50;
no_of_steps = 50;
time_step = 0.1;
rad = 0.05;

stoc = true;

interested_variable = 1;

fnameNN = 'Drone_NN.nt';

mid_point_init = zeros(12,1);
mid_point_init(1) = -0.0225;
mid_point_init(2) = 0.3089;
mid_point_init(3) = 0.2086;
mid_point_init(4) = 0.1407;
mid_point_init(5) = 0.6676;
mid_point_init(6) = 0.3038;
mid_point_init(7) = 0.0138;
mid_point_init(8) = -0.0110;
mid_point_init(9) = 0.0093;
mid_point_init(10) = -0.0366;
mid_point_init(11) = -0.0002;
mid_point_init(12) = -0.0002;

X = [];
Y = [];

simulation_count = 0;
while (simulation_count < no_of_traces )


  plot_time = zeros(1, no_of_steps+1);
  plot_variable = zeros(1, no_of_steps+1);

  % initial_state
  initial_state = zeros(12,1);
  initial_state(1) = mid_point_init(1) + (rad * rand);
  initial_state(2) = mid_point_init(2) + (rad * rand);
  initial_state(3) = mid_point_init(3) + (rad * rand);
  initial_state(4) = mid_point_init(4) + (rad * rand);
  initial_state(5) = mid_point_init(5) + (rad * rand);
  initial_state(6) = mid_point_init(6) + (rad * rand);
  initial_state(7) = mid_point_init(7) + (rad * rand);
  initial_state(8) = mid_point_init(8) + (rad * rand);
  initial_state(9) = mid_point_init(9) + (rad * rand);
  initial_state(10) = mid_point_init(10) + (rad * rand);
  initial_state(11) = mid_point_init(11) + (rad * rand);
  initial_state(12) = mid_point_init(12) + (rad * rand);

  x0 = initial_state;

  plot_time(1) = x0(interested_variable);
  plot_variable(1) = x0(interested_variable+1);
  i = 1;
  while (i <= no_of_steps)

    u = Drone_NN_output(x0, 100, 1, fnameNN);

    X = [X, x0];

    % ODE integration
    odefun = @(t,x) Drone_Dynamics(u, t, x );

    [ts, st] = ode45(odefun, 0:(time_step/50):time_step, x0);
    k = size(ts,1);

    if stoc
        x0 = st(k,:)' + 0.0025*randn(12, 1);
    else 
        x0 = st(k,:)';
    end

    i = i + 1;

    Y = [Y, x0];

  end

  simulation_count = simulation_count + 1;

end

if stoc
    save('Drone_stoc.mat', 'X', 'Y');
else
    save('Drone_det.mat', 'X', 'Y');
end

% This code is modified from original code written by Souradeep Dutta, taken
% from https://github.com/souradeep-111/sherlock, and licensed under the MIT
% license in the LICENSE_SHERLOCK file.

function [y] =  Drone_NN_output(x,offset,scale_factor,name)


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
    % i
    % weight_matrix
    % bias_matrix
    % g
end

y = g - offset * ones(no_of_outputs, 1);
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
