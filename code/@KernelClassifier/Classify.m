function results = Classify(obj, Y, Yt, varargin)
%CLASSIFY Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;
addRequired(p, 'Y');
addRequired(p, 'Yt');
parse(p, Y, Yt, varargin{:});

M = size(Y, 2);

% Compute the Gram matrix.
G = obj.compute_autocovariance(Y, obj.Sigma);

% W = inv((1/M)*G + obj.Lambda); %#ok<*MINV>
W = inv(G + M*obj.Lambda*eye(M)); %#ok<*MINV>

% Compute tau_n.
tau_n = 1 - min(diag((1/M)*G'*W*G)); 

% Compute Phi.
Phi = obj.compute_cross_covariance(Y, Yt, obj.Sigma);

% Classify points.
C = (1/M).*Phi'*W*Phi;
C_n = diag(C) >= 1 - tau_n;

% Output the results.
results.contains = C_n.';
results.classifier = @(x) (diag((1/M).* ...
    obj.compute_cross_covariance(Y, x, obj.Sigma)'* ...
    W*obj.compute_cross_covariance(Y, x, obj.Sigma)) >= 1 - tau_n).';

end

