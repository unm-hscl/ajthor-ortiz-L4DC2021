%% Define the problem.
load('figure4_samples.mat');

N = 200;

R = 50;     % Number of trajectories.
T = 200;    % Length of each trajectory.

% Create the figure for plotting.
df = figure('Units', 'points');
ax_data = axes(df);
ax_data.NextPlot = 'add';

results_CCO = SReachPoint(prob, alg_CCO, sys, X0);

%% Load system

load('figure4_samples.mat'); 

%% Plotting

% Generate output samples from initial condition.
M = R;       % number of observations.
Mt = 10000;     % Number of test points.

X0_traj = X(:, 1:T);

% Plot the trajectories.
for p = 1:R
    ph = plot(ax_data, X(1, (p-1)*200+1:(p-1)*200+200), X(2, (p-1)*200+1:(p-1)*200+200), 'r');
    ph.LineWidth = 0.5;
end

tic

for k = 1:T

%     X0_traj(:, k+1) = A*X0_traj(:, k) + B*U(:, k);
% 
%     X = A*X + B*U(:, k) + F*sys.Disturbance.sample(M);

    % Plot a random sample of the points.
    % scatter(ax_data, X(1, idx), X(2, idx), 'k.');

    % Generate test points.
    xt_xx = linspace(X0_traj(1, k) - 0.2, X0_traj(1, k) + 0.2, 100);
    yt_yy = linspace(X0_traj(2, k) - 0.2, X0_traj(2, k) + 0.2, 100);
    [XX, YY] = meshgrid(xt_xx, yt_yy);

    Xt = [
        reshape(XX, 1, []);
        reshape(YY, 1, []);
        repmat(X0_traj(3, k), [1 Mt]);
        repmat(X0_traj(4, k), [1 Mt]);
        ];

    %% Classify points.

    alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);
    results = alg.Classify(X(:,k:T:size(X,2)), Xt);

    C = double(reshape(results.contains, 100, 100));

    [~, ch] = contour(ax_data, xt_xx, yt_yy, C, [1 1]);
    ch.LineWidth = 1;
    ch.Color = 'b';

end

toc

ax_data.Title.String = '(a)';
ax_data.XLabel.Interpreter = 'latex';
ax_data.XLabel.String = '$z_{1}$';
ax_data.YLabel.Interpreter = 'latex';
ax_data.YLabel.String = '$z_{2}$';
ax_data.FontSize = 9;
