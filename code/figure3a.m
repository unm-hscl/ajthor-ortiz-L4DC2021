%% Define the problem.

N = 200;

R = 50;     % Number of trajectories.
T = 50;    % Length of each trajectory.

% Create the figure for plotting.
df = figure('Units', 'points', 'Position', [0, 0, 200, 150]);
ax_data = axes(df);
ax_data.NextPlot = 'add';

%% Load samples.

load('../data/Drone_det.mat');

%% Plotting

% Generate output samples from initial condition.
M = R;          % number of observations.
Mt = 10000;     % Number of test points.

X0_traj = X(:, 1:T);

% Plot the trajectories.
for p = 1:R
    ph = plot(ax_data, X(1, (p-1)*50+1:(p-1)*50+50), X(2, (p-1)*50+1:(p-1)*50+50), 'r');
    ph.LineWidth = 0.5;
end

%% Classify points.

alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);

tic

for k = 1:T

    % Generate test points.
    m1 = mean(X(1, k:T:size(X, 2)));
    m2 = mean(X(2, k:T:size(X, 2)));

    xt_xx = linspace(m1 - 0.25, m1 + 0.25, 100);
    yt_yy = linspace(m2 - 0.25, m2 + 0.25, 100);
    [XX, YY] = meshgrid(xt_xx, yt_yy);

    Xt = [
        reshape(XX, 1, []);
        reshape(YY, 1, []);
        repmat(mean(X(3, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(4, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(5, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(6, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(7, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(8, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(9, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(10, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(11, k:T:size(X, 2))), [1 Mt]);
        repmat(mean(X(12, k:T:size(X, 2))), [1 Mt]);
        ];

    results = alg.Classify(X(:, k:T:size(X, 2)), Xt);

    C = double(reshape(results.contains, 100, 100));

    [~, ch] = contour(ax_data, xt_xx, yt_yy, C, [1 1]);
    ch.LineWidth = 1;
    ch.Color = 'b';

end

toc

%% Plot the results.

ax_data.Title.String = '(a)';
ax_data.XLabel.Interpreter = 'latex';
ax_data.XLabel.String = '$x_{1}$';
ax_data.YLabel.Interpreter = 'latex';
ax_data.YLabel.String = '$x_{2}$';
ax_data.FontSize = 9;

saveas(gcf, '../results/figure3a','png');
saveas(gcf, '../results/figure3a','fig');
