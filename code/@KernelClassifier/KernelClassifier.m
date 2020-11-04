classdef KernelClassifier < srt.algorithms.Algorithm
% KERNELCLASSIFIER Kernel set classifier.
%
%   Kernel classifier.
%
%   Copyright 2019 Adam Thorpe
    
    properties (Access = private)

        % SIGMA_ Sigma parameter to Gaussian kernel.
        sigma_(1, 1) double {mustBeNumeric, mustBePositive} = 0.1
        % LAMBDA_ Regularization parameter.
        lambda_(1, 1) double {mustBeNumeric, mustBePositive} = 1

    end

    properties (Dependent)
        % SIGMA Gaussian kernel bandwidth parameter.
        Sigma
        % LAMBDA Regularization parameter.
        Lambda
    end
    
    methods
        function obj = KernelClassifier(varargin)
            %KERNELCLASSIFIER Construct an instance of this class
            %   Detailed explanation goes here

            % Call the parent constructor.
            obj = obj@srt.algorithms.Algorithm(varargin{:});

            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'sigma', 0.1);
            addParameter(p, 'lambda', 1);
            parse(p, varargin{:});

            obj.sigma_ = p.Results.sigma;
            obj.lambda_ = p.Results.lambda;

        end
    end

    % Static methods.
    methods (Static, Hidden)
        function n = compute_norm(x)
            % COMPUTE_NORM Compute the norm.
            M = size(x, 2);
            n = zeros(M);

            for k = 1:size(x, 1)
                n = n + (repmat(x(k, :), [M, 1]) - repmat(x(k, :)', [1, M])).^2;
                % G = G + abs(repmat(X(p, :), [M 1]) - repmat(X(p, :).', [1 M]));
            end
        end
        function n = compute_norm_cross(x, y)
            % COMPUTE_CROSS_NORM Compute the cross norm.
            M = size(x, 2);
            T = size(y, 2);

            n = zeros(M, T);

            for k = 1:size(x, 1)
                n = n + (repmat(y(k, :), [M, 1]) - repmat(x(k, :)', [1, T])).^2;
                % G = G + abs(repmat(Y(p, :), [M 1]) - repmat(X(p, :).', [1 T]));
            end
        end

        function cxx = compute_autocovariance(x, sigma)
            % COMPUTE_AUTOCOVARIANCECOMPUTE Compute autocovariance matrix.
            cxx = srt.algorithms.KernelEmbeddings.compute_norm(x);
            cxx = exp(-cxx/sigma);
        end
        function cxy = compute_cross_covariance(x, y, sigma)
            % COMPUTE_CROSS_COVARIANCE Compute cross-covariance matrix.
            cxy = srt.algorithms.KernelEmbeddings.compute_norm_cross(x, y);
            cxy = exp(-cxy/sigma);
        end
    end

    methods
        function set.Sigma(obj, sigma)
            validateattributes(sigma, {'double'}, {'positive', 'scalar'});
            obj.sigma_ = sigma;
        end
        function sigma = get.Sigma(obj)
            sigma = obj.sigma_;
        end
        function set.Lambda(obj, lambda)
            validateattributes(lambda, {'double'}, {'positive', 'scalar'});
            obj.lambda_ = lambda;
        end
        function lambda = get.Lambda(obj)
            lambda = obj.lambda_;
        end
    end
end

