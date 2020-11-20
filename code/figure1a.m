rng(5); % 1

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

%% Plotting

% Generate output samples from initial condition.
M = 100;       % number of observations.
Mt = 10000;     % Number of test points.

X = repmat(X0, [1 M]);

U = reshape(results_CCO.opt_input_vec, [2 N-1]);

X0_traj = zeros(4, N);
X0_traj(:, 1) = X0;

tic

for k = 1:N-1

    X0_traj(:, k+1) = A*X0_traj(:, k) + B*U(:, k);

    X = A*X + B*U(:, k) + F*sys.Disturbance.sample(M);

    % Plot a random sample of the points.
    % scatter(ax_data, X(1, idx), X(2, idx), 'k.');

    % Generate test points.
    xt_xx = linspace(X0_traj(1, k+1) - 0.2, X0_traj(1, k+1) + 0.2, 100);
    yt_yy = linspace(X0_traj(2, k+1) - 0.2, X0_traj(2, k+1) + 0.2, 100);
    [XX, YY] = meshgrid(xt_xx, yt_yy);

    Xt = [
        reshape(XX, 1, []);
        reshape(YY, 1, []);
        repmat(mean(X(3, :)), [1 Mt]);
        repmat(mean(X(4, :)), [1 Mt])
        ];

    %% Classify points.

    alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);
    results = alg.Classify(X, Xt);

    C = double(reshape(results.contains, 100, 100));

    [~, ch] = contour(ax_data, xt_xx, yt_yy, C, [1 1]);
    ch.LineWidth = 1;
    ch.Color = 'b';

end

toc

% Plot the unperturbed trajectory.
ph = plot(ax_data, X0_traj(1, :), X0_traj(2, :), 'rx-');
ph.LineWidth = 1;

X0 = [-0.75; -0.75; 0; 0];
ph = plot(ax_data, X0(1), X0(2), 'rx-');
ph.LineWidth = 1;

ax_data.Title.String = '(a)';
ax_data.XLabel.Interpreter = 'latex';
ax_data.XLabel.String = '$z_{1}$';
ax_data.YLabel.Interpreter = 'latex';
ax_data.YLabel.String = '$z_{2}$';
ax_data.FontSize = 9;
