% Demostration of GreedyNAR, NAR, stochastic collocation on borehole
% function.

%% prepare data

clear
rng(2022)

rw = 0.1;
r  = 20000;
Tu = 100000;
Hu = 1000;
Tl = 100;
Hl = 760;
L  = 1200;
Kw = 10000;

func_hf = @ (rw) borehole([rw, r, Tu, Hu, Tl, Hl, L, Kw]);
func_lf = @ (rw) boreholelc([rw, r, Tu, Hu, Tl, Hl, L, Kw]);

% xte = 100:100:50000;
xte = [0.05:0.001:0.15]';

Ntr = 20;
xtr = rand(Ntr,1) * 0.1 + 0.05;

for i = 1:length(xte)
    yhf_te(i,1) =  func_hf(xte(i));
    ylf_te(i,1) =  func_lf(xte(i));
end
for i = 1:length(xtr)
    yhf_tr(i,1) =  func_hf(xtr(i));
    ylf_tr(i,1) =  func_lf(xtr(i));
end


figure(1)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')

plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')
hold off

%% normal GP with high-fidelity data only
% model_hf = cigp(xtr, yhf_tr, xte, 'ard');
% model_hf = cigp(xtr, yhf_tr, xte, 'linear');
model_hf = cigp_sequen(xtr, yhf_tr, xte, 'ard', 10);

figure(2)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')
plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')

plot(xte,model_hf.yTe_pred,'r-')
hold off

%% stochastic collocation
[yPred, model,idx_abs] = sc_v01(xtr,{ylf_tr,yhf_tr},xte,[Ntr,Ntr]);
figure(3)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')
plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')

plot(xte,yPred,'r-')

hold off

%% stochastic collocation with low-fidelity observations
[yPred, model,idx_abs] = sc_v02(xtr,{ylf_tr,yhf_tr},xte,{ylf_te,[]},[Ntr,Ntr/2]);
figure(4)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')
plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')

plot(xte,yPred,'r--')

hold off

%%
[yPred, model, absId_use] = GreedyNAR(xtr,{ylf_tr,yhf_tr},xte,'ard',[Ntr,Ntr/2]);

figure(4)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')
plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')

plot(xte,yPred,'r--')

hold off


%%
[yPred, model, absId_use] = GreedyNAR_v02(xtr,{ylf_tr,yhf_tr},xte,{ylf_te,[]},'ard',[Ntr,Ntr/2]);

figure(4)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ylf_te,'g-')
plot(xtr,yhf_tr,'ko')
plot(xtr,ylf_tr,'go')

plot(xte,yPred,'r--')

hold off


