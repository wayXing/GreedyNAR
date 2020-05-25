function [yPred, model,idx_abs] = sc_v01(xtr,Ytr,xte,ntr)
% multi-fidelity stochastic collation version 01
% Note: the performance is generally poor because SC based method requires
%       observation on low-fidelity. 
% logg: v01 uses no low-fidelity groundtruth data
% 
% Inputs:
% xtr - [N_train x dim_x] matrix, input parameters
% Ytr - [1 x N_fidelity] cell, each element contains the corresponding output to
%       xtr and has to be a [N_train x dim_y] matrix. Note that not all data
%       would be used for training
% xte - [N_test x dim_x] matrix, testing inputs 
% ntr - [1 x N_fidelity] array, indicating how many training points are provided
%       for each fidelity. The first-fidelity use the first N training samples
%       for training. The reset is based on a greedy selection.
% 
% Outputs:
% yPred - predictions for xte at the highest-fidelity
% model - model info
% idx_abs - important index for all training data 
% 
% Author: Wei Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 21-May-2020
% 
%% initial
coreGp_func = @colocation;  
% yhPred = colocation(yl,yh,yl_star)
% assert(size(xtr,1)==size(Ytr{1},1),'first fidelity n not equal to xtr')
nFidelity = length(Ytr);    

%% main
%initial training data choice for F1 (fidelity-1)
idx{1} = 1:1:ntr(1);   %no ordering infomation for idx{1}
for f = 1:nFidelity
    
    if f <= 1
        %no prior idx knowledge thus use given sequence
        id_use_f = 1:ntr(f);
        ytr_lastF = xtr(id_use_f,:);
        yPred_lastF = xte;
    else
        id_use_f = idx_abs{f-1}(1:ntr(f));
        ytr_lastF = Ytr{f-1}(id_use_f,:);
        yPred_lastF = Ypred{f-1};
    end
        
    ytr_f = Ytr{f}(id_use_f,:);  % the current fidelity y
    K{f} = ytr_f*ytr_f';     %gram matrix for current (f) fidelity data

    [~, ~, p, q] = gecp(K{f});
    for i = 1: size(q,1) 
%           idx{f}(i) = find(p(i,:)==1);
%         idx{f}(i) = find(q(:,i)==1);
          idx{f}(i) = find(p(i,:)==1);     % the importance idx for id_use_f
    end

    Ypred{f} = colocation(ytr_lastF, ytr_f, yPred_lastF);
    idx_abs{f} = id_use_f(idx{f});    

end

model.K = K;
model.Ypred = Ypred;

yPred = Ypred{end};
end


