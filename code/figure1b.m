%% Define the problem.
N = 5;

ymax = 1;
vxmax = 0.5;
vymax = 0.5;

A_safe_set = [
     1,  1,  0,  0;
    -1,  1,  0,  0;
     0, -1,  0,  0;
     0,  0,  1,  0;
     0,  0, -1,  0;
     0,  0,  0,  1;
     0,  0,  0, -1
    ];

b_safe_set = [
     0;
     0;
     ymax;
     vxmax;
     vxmax;
     vymax;
     vymax
    ];

safe_set = Polyhedron(A_safe_set, b_safe_set);

target_set = Polyhedron('lb', [-0.1; -0.1; -0.01; -0.01], ...
                        'ub', [ 0.1;    0;  0.01;  0.01]);

T = srt.Tube(N, safe_set);
T.tube(N) = target_set;

prob = srt.problems.TerminalHitting( ...
    'TargetTube', T, ...
    'ConstraintTube', T);

safe_set_projection = safe_set.slice([3, 4], zeros(2, 1));
target_set_projection = target_set.slice([3, 4], zeros(2, 1));

% Create the figure for plotting.
df = figure('Units', 'points', 'Position', [0, 0, 200, 150]);
ax_data = axes(df);
ax_data.NextPlot = 'add';
plot(safe_set_projection, 'color', 'y', 'alpha', 0.1);
plot(target_set_projection, 'color', 'g', 'alpha', 0.1);

%% Define the system.

Ts = 20;

sys = srtCWHModel(Ts, ...
    'Dimensionality', 4);

A = sys.A;
B = sys.B;
F = sys.F;

% Specify the initial condition x0.
X0 = [-0.75; -0.75; 0; 0];

% Generate optimal control policy from initial condition.
alg_CCO = srt.algorithms.ChanceOpen('pwa_accuracy', 1E-3);

results_CCO = SReachPoint(prob, alg_CCO, sys, X0);

U = reshape(results_CCO.opt_input_vec, [2 N-1]);

X0_traj = zeros(4, N);
X0_traj(:, 1) = X0;

tic

for k = 1:N-1

    X0_traj(:, k+1) = A*X0_traj(:, k) + B*U(:, k);

end

RV = srt.SReachFwd('concat-stoch', sys, X0, N);
Mu = RV.Mean();
Sigma =  RV.Sigma();

for k = 2:N

    idx = [(k - 1)*4+1, (k - 1)*4+2];

    Cxy = Sigma(idx, idx);

    x0 = X0_traj(1, k);
    y0 = X0_traj(2, k);

    [x, y, ~] = get_points(Cxy);
    h1 = plot(x0 + k*x, y0 + k*y, 'b');
    h1.LineWidth = 1;

end

%% Plotting

% Plot the unperturbed trajectory.
ph = plot(ax_data, X0_traj(1, :), X0_traj(2, :), 'rx-');
ph.LineWidth = 1;

ax_data.Title.String = '(b)';
ax_data.XLabel.Interpreter = 'latex';
ax_data.XLabel.String = '$z_{1}$';
ax_data.YLabel.Interpreter = 'latex';
ax_data.YLabel.String = '$z_{2}$';
ax_data.FontSize = 9;


function [x,y,z] = get_points(C,clipping_radius)
n=100; % Number of points around ellipse
p=0:pi/n:2*pi; % angles around a circle
[V, D] = eig(C); % Compute eigen-stuff
xy = [cos(p'),sin(p')] * sqrt(D) * V'; % Transformation
x = xy(:,1);
y = xy(:,2);
z = zeros(size(x));
% Clip data to a bounding radius
if nargin >= 2
  r = sqrt(sum(xy.^2,2)); % Euclidian distance (distance from center)
  x(r > clipping_radius) = nan;
  y(r > clipping_radius) = nan;
  z(r > clipping_radius) = nan;
end
end
