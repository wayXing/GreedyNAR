function yhPred = colocation(yl,yh,yl_star)
% collocation method for interpolation
% 
% yl: low-fidelity y
% yh: high-fidelity y
% y1_star: test 
% 
% Author: Wei Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 21-May-2020
% 
%%
% applying normalization before algorithm
% [yl,ylmean,ylstd] = normal(yl);
% [yh,yhmean,yhstd] = normal(yh);

assert(size(yl,1)==size(yh,1),'training sampe do not match')

Gl = yl * yl';  %gramian matrix low-fidelity y 
Gh = yh * yh';  %gramian matrix high-fidelity y

Gl = Gl + eye(size(Gl)).* 1e-6;     %adding jitter to stablize the calculation. !IMPORTANT
Gh = Gh + eye(size(Gh)).* 1e-6;     

%different method for solving the inverse system
% c = linsolve(Gl, yl * yl_star');
% c = inv(Gl) * yl * yl_star';
c = pinv(Gl) * yl * yl_star';     % the only method that works well
% c = Gl \ (yl * yl_star');
% c = eigenSolve(Gl, yl * yl_star');

yhPred = c' * yh;
% yhPred = yhPred .* yhstd + yhmean;

end


function [y,ymean,ystd] = normal(y)
ymean = mean(y(:));
ystd = std(y(:));
y = (y-ymean)./ystd;
end

function x = eigenSolve(A,b)
tol = 1e-4;

[u,s,v] = svd(A);
invs = diag(1./(diag(s)+tol));
id = diag(s)>tol;
x = u(id,id)*invs(id,id)*v(id,id)'*b;

end
