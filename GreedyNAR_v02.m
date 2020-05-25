function [yPred, model, absId_use] = GreedyNAR_v02(xtr,Ytr,xte,Yte,kernel,ntr,option)
% GreedyNAR 
% 
% logg: v01 uses no low-fidelity groundtruth data
%       v02 uses available low-fidelity data
% 
% Inputs:
% xtr - [N_train x dim_x] matrix, input parameters
% Ytr - [1 x N_fidelity] cell, each element contains the corresponding output to
%       xtr and has to be a [N_train x dim_y] matrix. Note that not all data
%       would be used for training
% xte - [N_test x dim_x] matrix, testing inputs 
% Yte - [1 x N_fidelity] cell, each element contains the corresponding output to
%       xte and has to be a N_test x dim_y matrix. OR the cell can be empty ([])
%       to provide no low-fidelity information
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
%% initial
if nargin < 7
   option.step = 1; %the step of increasing the training data
   option.initial_size = 2; %the initial size of training data
end
coreGp_func = @cigp_sequen;  
nFidelity = length(Ytr); 

%% main
%initial training data choice for F1 (fidelity-1)
absId_use_0 = 1:1:size(xtr,1);

f=1;
Model{f} = coreGp_func(xtr, Ytr{f}, xte, kernel, ntr(f), option);
absId_use{f} = absId_use_0(Model{f}.id_use);

for f = 2:nFidelity

    xte_f = [xte, Model{f-1}.yTe_pred];
    if ~isempty(Yte{f-1})
        xte_f = [xte, Yte{f-1}];    %overwrite yPred_lastF using avalible observations   
    end
    
    xtr_f = [xtr(absId_use{f-1},:) ,Ytr{f-1}(absId_use{f-1},:)];

    Model{f} = coreGp_func(xtr_f, Ytr{f}(absId_use{f-1},:), xte_f, kernel, ntr(f), option);
    absId_use{f} = absId_use{f-1}(Model{f}.id_use);
end

%another concatination
% for f = 2:nFidelity
% 
%     xte_f = [Model{f-1}.yTe_pred];
%     if ~isempty(Yte{f-1})
%         xte_f = [Yte{f-1}];    %overwrite yPred_lastF using avalible observations   
%     end
%     
%     xtr_f = [Ytr{f-1}(absId_use{f-1},:)];
% 
%     Model{f} = coreGp_func(xtr_f, Ytr{f}(absId_use{f-1},:), xte_f, kernel, ntr(f), option);
%     absId_use{f} = absId_use{f-1}(Model{f}.id_use);
% end



yPred = Model{nFidelity}.yTe_pred;
model.submodels = Model;
model.absId_use = absId_use;

end