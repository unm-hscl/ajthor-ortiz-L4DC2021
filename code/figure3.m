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
df = figure;
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

% Specify the terminal condition xN.
XN = [0; 0; 0; 0];

% Generate optimal control policy from initial condition.
alg_CCO = srt.algorithms.ChanceOpen('pwa_accuracy', 1E-3);

alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);

% Generate samples.
M = 1000;       % number of observations.
Mt = 1024;     % Number of test points.

x0_xx = linspace(-1, 1, 32);
y0_yy = linspace(-1, 0, 32);
[XX, YY] = meshgrid(x0_xx, y0_yy);
X0 = [
    reshape(XX, 1, []);
    reshape(YY, 1, []);
    zeros(1, Mt);
    zeros(1, Mt)
    ];

C = zeros(1, size(X0, 2));

for n = 1:Mt

    % If the state does not meet constraints, no open loop policy will be found.
    if ~T.contains(1, X0(:, n))
        continue;
    end

    % Compute the open loop control sequence from this initial condition.
    results_CCO = SReachPoint(prob, alg_CCO, sys, X0(:, n));

    % If the result is nan, meaning no open loop policy was found, move on.
    if any(isnan(results_CCO.opt_input_vec))
        continue;
    end

    % Compute the terminal states from the initial condition.
    X = repmat(X0(:, n), [1 M]);

    U = reshape(results_CCO.opt_input_vec, [2 N-1]);

    for k = 1:N-1
        X = A*X + B*U(:, k) + F*sys.Disturbance.sample(M);
    end

    % Classify whether the terminal condition is in the support.
    results = alg.Classify(X, XN);

    C(n) = double(results.contains);

end

%% Plot the results.

C = reshape(C, 32, 32);

[~, ch] = contour(ax_data, x0_xx, y0_yy, C, [1 1]);
ch.LineWidth = 1;
ch.Color = 'b';
