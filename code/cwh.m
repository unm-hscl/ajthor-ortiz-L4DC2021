
%% Define the problem.
N = 5;

ymax = 2;
vxmax = 0.5;
vymax = 0.5;

A_safe_set = [1, 1, 0, 0;
             -1, 1, 0, 0;
              0, -1, 0, 0;
              0, 0, 1,0;
              0, 0,-1,0;
              0, 0, 0,1;
              0, 0, 0,-1];

b_safe_set = [0;
              0;
              ymax;
              vxmax;
              vxmax;
              vymax;
              vymax];

safe_set = Polyhedron(A_safe_set, b_safe_set);

target_set = Polyhedron('lb', [-0.1; -0.1; -0.01; -0.01], ...
                        'ub', [ 0.1;    0;  0.01;  0.01]);

T = srt.Tube(N, safe_set);
T.tube(N) = target_set;

prob = srt.problems.TerminalHitting( ...
    'TargetTube', T, ...
    'ConstraintTube', T);

%% Define the system.

Ts = 20;

sys = srtCWHModel(Ts, ...
    'Dimensionality', 4);

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

A = sys.A;
B = sys.B;
F = sys.F;

for k = 1:N-1
    X = A*X + B*U(:, k) + F*sys.Disturbance.sample(M);
end

%% Create set classifier.
%

sigma = 0.1;
lambda = 1/M;

% Compute the Gram matrix.
G = zeros(M);

for p = 1:size(X, 1)
    G = G + (repmat(X(p, :), [M 1]) - repmat(X(p, :).', [1 M])).^2;
%     G = G + abs(repmat(X(p, :), [M 1]) - repmat(X(p, :).', [1 M]));
end

G = exp(-sqrt(G)./sigma);

% W = inv((1/M)*G + lambda); %#ok<*MINV>
W = inv(G + M*lambda*eye(M)); %#ok<*MINV>

tau_n = 1 - min(diag((1/M)*G'*W*G));

%% Classify points.
% 

% Generate test points. 

Mt = 10;

% Points probably inside. 
Xt = [ 
    -0.1 + 0.2*rand(1, Mt);
    -0.1 + 0.1*rand(1, Mt);
    -0.01 + 0.02*rand(1, Mt);
    -0.01 + 0.02*rand(1, Mt);
    ];

% Point definitely outside.
% Xt(:, end) = [0; -1; 0; 0]; 

% Xt = X0;

% Compute Phi.
Phi = zeros(M, Mt);

for p = 1:size(X, 1)
    Phi = Phi + (repmat(Xt(p, :), [M 1]) - repmat(X(p, :).', [1 Mt])).^2;
%     Phi = Phi + abs(repmat(Xt(p, :), [M 1]) - repmat(X(p, :).', [1 Mt]));
end

Phi = exp(-sqrt(Phi)./sigma);
% Phi = Phi./sum(abs(Phi), 1);

% Classify points.
C = (1/M).*Phi'*W*Phi;

C_n = diag(C) >= 1 - tau_n;


alg = KernelClassifier('sigma', 0.1, 'lambda', 1/M);
results = alg.Classify(X, Xt);

%% Plot the results.
% 

%%

d.name = 'Classifier';
F = Function(results.classifier, d);

safe_set_intersection = safe_set.addFunction(F, 'alpha');
safe_set_intersection_projection = safe_set_intersection.slice([3, 4], zeros(2, 1));

safe_set_projection = safe_set.slice([3, 4], zeros(2, 1));
target_set_projection = target_set.slice([3, 4], zeros(2, 1));

figure
hold on
plot(safe_set_projection, 'color', 'y', 'alpha', 0.1);
plot(target_set_projection, 'color', 'g', 'alpha', 0.1);
hold off

