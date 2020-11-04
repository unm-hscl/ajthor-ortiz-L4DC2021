
plotting = true;

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

if plotting
    df = figure;
    ax_data = axes(df);
    ax_data.NextPlot = 'add';
    plot(safe_set_projection, 'color', 'y', 'alpha', 0.1);
    plot(target_set_projection, 'color', 'g', 'alpha', 0.1);

    sf = figure;
    ax_surf = axes(sf);
    ax_surf.NextPlot = 'add';
    plot(safe_set_projection, 'color', 'y', 'alpha', 0.1);
    plot(target_set_projection, 'color', 'g', 'alpha', 0.1);

    gf = figure;
    ax_param = axes(gf);
    ax_param.NextPlot = 'add';
    plot(safe_set_projection, 'color', 'y', 'alpha', 0.1);
    plot(target_set_projection, 'color', 'g', 'alpha', 0.1);

end

%% Define the system.

Ts = 20;

sys = srtCWHModel(Ts, ...
    'Dimensionality', 4);

A = sys.A;
B = sys.B;
F = sys.F;

%% Generate samples.
%

% Specify the initial xondition x0.
X0 = [-0.75; -0.75; 0; 0];

% Generate optimal control policy from initial condition.
alg_CCO = srt.algorithms.ChanceOpen('pwa_accuracy', 1E-3);

results_CCO = SReachPoint(prob, alg_CCO, sys, X0);

% Generate output samples from initial condition.
M = 1000;

X = repmat(X0, [1 M]);

U = reshape(results_CCO.opt_input_vec, [2 N-1]);

% Plot a random sample of points from the data.
idx = randperm(M, 50);

for k = 1:N-1
    X = A*X + B*U(:, k) + F*sys.Disturbance.sample(M);

    if plotting, scatter(ax_data, X(1, idx), X(2, idx), 'k.'); end
end

% Plot the unperturbed trajectory.
if plotting
    X0_traj = zeros(4, N);
    X0_traj(:, 1) = X0;

    for k = 1:N-1
        X0_traj(:, k+1) = A*X0_traj(:, k) + B*U(:, k);
    end

    ph = plot(ax_data, X0_traj(1, :), X0_traj(2, :), 'rx-');
    ph.LineWidth = 1;
end

% Plot the terminal states.
if plotting, scatter(ax_surf,  X(1, :), X(2, :), 'k.'); end
if plotting, scatter(ax_param, X(1, :), X(2, :), 'k.'); end

% Generate test points.

Mt = 10000;

xt_xx = linspace(-0.2, 0.2, 100);
yt_yy = linspace(-0.2, 0.1, 100);
[XX, YY] = meshgrid(xt_xx, yt_yy);

Xt = [
    reshape(XX, 1, []);
    reshape(YY, 1, []);
    zeros(1, Mt);
    zeros(1, Mt)
    ];

%% Classify points.

alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);
results = alg.Classify(X, Xt);

C = double(reshape(results.contains, 100, 100));

if plotting
    ax_surf.XLim = [-0.2, 0.2];
    ax_surf.YLim = [-0.2, 0.1];
    [~, ch] = contour(ax_surf, xt_xx, yt_yy, C, [1 1]);
    ch.LineWidth = 1;
    ch.Color = 'r';
end

%% Classify points for different values of sigma.

lambdas = [1/M, 1/(M^2), 1/(M^3)];

for p = 1:length(lambdas)

    alg = KernelClassifier('sigma', 0.1, 'lambda', lambdas(p));
    results_P = alg.Classify(X, Xt);

    C = double(reshape(results_P.contains, 100, 100));

    if plotting
        ax_param.XLim = [-0.2, 0.2];
        ax_param.YLim = [-0.2, 0.1];
        [~, ch] = contour(ax_param, xt_xx, yt_yy, C, [1 1]);
        ch.LineWidth = 1;
        ch.Color = 'r';

        ch0 = get(ch, 'children');
        set(ch0, 'facealpha', 0.1);

%         if sigmas(s) == 0.1
%             ch.Alpha = 1;
%         else
%             ch.Alpha = 0.1;
%         end
    end

end
