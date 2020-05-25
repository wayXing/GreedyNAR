function model = cigp_sequen(xTr, yTr, xTe, kernel, ntr, option)
% conditional independet (i.e., coregionalization matrix = I) GP for
% multivariate output with sequencial learning
% WeiXing. May 2020
% 
% kernel = 'ard' or 'linear'
% 
% logg: v1: initial version. Default normalization
%       seque: with sequential learning. Not all xTr/yTr would be used 


% sequential setting
if nargin < 6
   option.step = 1; %the step of increasing the training data
   option.initial_size = 5; %the initial size of training data
end
% optimizer setting
opt = [];
opt.MaxIter = 100;
opt.MaxFunEvals = 10000;

    %% default using normalization for x and y
    nSample_tr = size(xTr,1);
    nSample_te = size(xTe,1);
    % dimX = 

    %for x 
    %seperate each dim
    meanX = mean(xTr);
    stdX = std(xTr);
    %combine each dim
%     meanX = repmat(mean(xTr(:)), 1,size(xTr,2));
%     stdX = repmat(std(xTr(:)), 1,size(xTr,2));
    %normalize data
    xTr = (xTr - repmat(meanX, nSample_tr, 1) ) ./ (repmat(stdX, nSample_tr, 1) + eps);
    xTe = (xTe - repmat(meanX, nSample_te, 1) ) ./ (repmat(stdX, nSample_te, 1) + eps);

    %for y 
    %seperate each dim 
%     meanY = mean(yTr);
%     stdY = std(yTr);
    %combine each dim
    meanY = repmat(mean(yTr(:)), 1,size(yTr,2));
    stdY = repmat(std(yTr(:)), 1,size(yTr,2));
    %normalize data(
    yTr = (yTr - repmat(meanY,nSample_tr,1) ) ./ (repmat(stdY,nSample_tr,1) + eps);

    %% add noise for model stability (prevent beta from going to inf)
    %     yTr = yTr + randn(size(yTr)) ./ 100;
    
    %% init parameters
    assert(size(xTr,1)==size(yTr,1),'inconsistent data');
    [N,d] = size(xTr);
    
    id_use = 1:option.initial_size;
    id_candidate = setdiff(1:nSample_tr,id_use);
    
%     log_bta = log(1/var(yTr(:)));
    log_bta = log(1/1e-4);
%     log_bta = log(1/eps);
    log_l = zeros(d,1);
    %log_l = 2*log(median(pdist(Xtr)))*ones(d,1);
    log_sigma = 0;
%     log_sigma0 = log(1e-4);
    log_sigma0 = log(eps);
    params = [log_l;log_sigma;log_sigma0;log_bta];
    a0 = 1e-3; b0 = 1e-3;   %parameter for the noise prior, Gamma
    
    %% main
    while length(id_use) < ntr
        
        ixTr = xTr(id_use,:);
        iyTr = yTr(id_use,:);
        
        [iN,d] = size(ixTr);
        m = size(iyTr,2);
        D = iyTr*iyTr'; 
        
        %fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, kernel), params);
        %max_iter = 1000;
        %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
        new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, ixTr, D, kernel), params, opt);
%         params = new_params;
        
        [ker_param,idx] = load_kernel_parameter(new_params, d, kernel, 0);
        bta = exp(new_params(idx+1));
        
        %candidate pred
        xTr_candiddate = xTr(id_candidate,:);
        
        Sigma = 1/bta*eye(iN) + ker_func(ixTr,ker_param);
        Ksn = ker_cross(xTr_candiddate, ixTr, ker_param);
        
%         mean_candidate = Knn*(Sigma\iyTr);
        var_candidate = diag(ker_cross(xTr_candiddate,xTr_candiddate,ker_param)) - diag(Ksn*(Sigma\Ksn'));
%         yTe_var = var_candidate + 1/bta;

        [~, I] = sort(var_candidate,'descend');
        
        import_list = id_candidate(I);
        
        % update id_use
        if length(id_use) + option.step > ntr %last update if condition is met
            id_use = [id_use, import_list(1: ntr - length(id_use) )];
            
            ixTr = xTr(id_use,:);
            iyTr = yTr(id_use,:);
            [iN,d] = size(ixTr);
            m = size(iyTr,2);
            D = iyTr*iyTr'; 

            new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, ixTr, D, kernel), params, opt);
            params = new_params;
        else
            id_use = [id_use, import_list(1: option.step)];
        end
        id_candidate = setdiff(1:nSample_tr,id_use);
        
        
    end
    
    %% save
    [ker_param,idx] = load_kernel_parameter(params, d, kernel, 0);
    bta = exp(params(idx+1));
    model = [];
    model.params = params;
    model.ker_param = ker_param;
    model.bta = bta;
    model.id_use = id_use;
        
    %% make predictions for xte/xtr
    ixTr = xTr(id_use,:);
    iyTr = yTr(id_use,:);

    [iN,d] = size(ixTr);
%     m = size(iyTr,2);
%     D = iyTr*iyTr'; 
    
    %tr pred
    Sigma = 1/bta*eye(iN) + ker_func(ixTr,ker_param);
    Knn = ker_cross(xTr, ixTr, ker_param);
    yTr_pred = Knn*(Sigma\iyTr);
    %de-normalize
    yTr_pred = yTr_pred .* repmat(stdY,size(yTr_pred,1),1) + repmat(meanY,size(yTr_pred,1),1);
    model.yTr_pred = yTr_pred;
    
    %te pred
    if ~isempty(xTe)
        Ksn = ker_cross(xTe,ixTr,ker_param);
        yTe_pred = Ksn*(Sigma\iyTr);
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;
        
        %de-normalize
        yTe_pred = yTe_pred .* repmat(stdY,size(yTe_pred,1),1) + repmat(meanY,size(yTe_pred,1),1);
        model.yTe_pred = yTe_pred;
        
        model.fTe_var = repmat(fTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
        model.yTe_var = repmat(yTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
        
    end
    
%     xTr = (xTr - repmat(meanX,size(xTr,1)) ) ./ repmat(stdX,size(xTr,1));
%     yTr = (yTr - repmat(meanY,size(yTr,1)) ) ./ repmat(stdY,size(yTr,1));
%     xTe = (xTe - repmat(meanX,size(xTe,1)) ) ./ repmat(stdX,size(xTe,1));


    model.forward = @forward;   %prediction function
    %save the predictive function
    function [yTe_pred,fTe_var]=forward(xTe)
        
        xTe = (xTe - repmat(meanX,size(xTe,1),1) ) ./ (repmat(stdX,size(xTe,1),1) + eps);

        Ksn = ker_cross(xTe,ixTr,ker_param);
        yTe_pred = Ksn*(Sigma\iyTr);
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;

        %de-normalize
        yTe_pred = yTe_pred .* repmat(stdY,size(yTe_pred,1),1) + repmat(meanY,size(yTe_pred,1),1);       
    %     model.yTe_pred = yTe_pred;

        fTe_var = repmat(fTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
%         yTe_var = repmat(yTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
    end

    
end






