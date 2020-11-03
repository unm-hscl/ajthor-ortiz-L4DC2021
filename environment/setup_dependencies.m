%% YALMIP Install
%
addpath(genpath('/YALMIP'));
disp('Installed YALMIP');

%% CVX Install
%
cd('/cvx');
evalc('cvx_setup()');
disp('Installed CVX (Standard bundle)');

%% MPT3 Install
%
mkdir('/tbxmanager');
cd('/tbxmanager');
urlwrite('http://www.tbxmanager.com/tbxmanager.m', 'tbxmanager.m');
a=evalc('tbxmanager');
disp('Installed tbxmanager');

tbxmanager install mpt mptdoc cddmex fourier glpkmex hysdel lcp sedumi espresso

evalc('mpt_init');

a = mptopt('lpsolver','glpk','qpsolver','quadprog');

disp('Installed MPT3');

%% SReachTools Install
%
addpath(genpath('/SReachTools'));
disp('Installed SReachTools');


savepath; % For compatibility with CodeOcean.
