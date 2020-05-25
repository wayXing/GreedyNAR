% Demostration of GreedyNAR, NAR, stochastic collocation on synthetic 
% three-fidelity data

%% prepare data

clear
rng(202)

func_lf = @ (x) exp(-x) .* sin(x)
func_mf = @ (x) exp(-x) .* sin(x) .* cos(x)
func_hf = @ (x) exp(-x) .* sin(x) .* cos(x) + tanh(x) * 0.05
% xte = 100:100:50000;
xte = [0:0.01:pi*2]';

Ntr = 32;
% xtr = rand(Ntr,1) * pi * 2
xtr = linspace(0, pi * 2, Ntr)';

for i = 1:length(xte)
    yhf_te(i,1) =  func_hf(xte(i));
    ymf_te(i,1) =  func_mf(xte(i));
    ylf_te(i,1) =  func_lf(xte(i));
end
for i = 1:length(xtr)
    yhf_tr(i,1) =  func_hf(xtr(i));
    ymf_tr(i,1) =  func_mf(xtr(i));
    ylf_tr(i,1) =  func_lf(xtr(i));
end


figure(1)
plot(xte,yhf_te,'k-')
hold on 
plot(xte,ymf_te,'m-')
plot(xte,ylf_te,'g-')

plot(xtr,yhf_tr,'ko')
plot(xtr,ymf_tr,'mo')
plot(xtr,ylf_tr,'go')
hold off



%% normal GP with high-fidelity data only
option.step = 1; %the step of increasing the training data
option.initial_size = 2; %the initial size of training data

% model_hf = cigp(xtr, yhf_tr, xte, 'ard');
% model_hf = cigp(xtr, yhf_tr, xte, 'linear');
model_hf = cigp_sequen(xtr, yhf_tr, xte, 'ard', Ntr/4,option);
% model_hf = cigp(xtr(1:Ntr/4), yhf_tr(1:Ntr/4), xte, 'ard');


figure(2)
clf
hold on 
plot(xte,ylf_te,'g-','LineWidth',2, 'MarkerSize',10)
plot(xte,ymf_te,'m-','LineWidth',2, 'MarkerSize',10)
plot(xte,yhf_te,'b-','LineWidth',2, 'MarkerSize',10)

plot(xte,model_hf.yTe_pred,'r--','LineWidth',2, 'MarkerSize',10)

id_use = model_hf.id_use;
plot(xtr(id_use),yhf_tr(id_use),'b+','LineWidth',2, 'MarkerSize',10)
hold off

legend('Low-fidelity groundtruth','Mid-fidelity groundtruth','High-fidelity groundtruth','High-fidelity prediction',...
 'High-fidelity used observations')
box on 
grid on 
set(gca,'FontSize',18);

print('IMG/gp_f3','-dpng')

%% stochastic collocation
[yPred, model,absId_use] = sc_v01(xtr,{ylf_tr,ymf_tr,yhf_tr},xte,[Ntr,Ntr/2,Ntr/4]);
figure(3)
clf
hold on 
plot(xte,ylf_te,'g-','LineWidth',2, 'MarkerSize',10)
plot(xte,ymf_te,'m-','LineWidth',2, 'MarkerSize',10)
plot(xte,yhf_te,'b-','LineWidth',2, 'MarkerSize',10)

plot(xte,yPred,'r--','LineWidth',2, 'MarkerSize',10)

% id_use = absId_use{3};
plot(xtr(absId_use{1}),ylf_tr(absId_use{1}),'g+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{2}),ymf_tr(absId_use{2}),'m+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{3}),yhf_tr(absId_use{3}),'b+','LineWidth',2, 'MarkerSize',10)
hold off

legend('Low-fidelity groundtruth','Mid-fidelity groundtruth','High-fidelity groundtruth','High-fidelity prediction',...
    'Low-fidelity used observations','Mid-fidelity used observations','High-fidelity used observations')
box on 
grid on 
set(gca,'FontSize',18);

print('IMG/sc1_fx3','-dpng')

%% stochastic collocation with low-fidelity observations
[yPred, model,absId_use] = sc_v02(xtr,{ylf_tr,ymf_tr,yhf_tr},xte,{ylf_te,[],[]},[Ntr,Ntr/2,Ntr/4]);
figure(4)
clf
hold on 
plot(xte,ylf_te,'g-','LineWidth',2, 'MarkerSize',10)
plot(xte,ymf_te,'m-','LineWidth',2, 'MarkerSize',10)
plot(xte,yhf_te,'b-','LineWidth',2, 'MarkerSize',10)

plot(xte,yPred,'r--','LineWidth',2, 'MarkerSize',10)

% id_use = absId_use{3};
plot(xtr(absId_use{1}),ylf_tr(absId_use{1}),'g+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{2}),ymf_tr(absId_use{2}),'m+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{3}),yhf_tr(absId_use{3}),'b+','LineWidth',2, 'MarkerSize',10)
hold off

legend('Low-fidelity groundtruth','Mid-fidelity groundtruth','High-fidelity groundtruth','High-fidelity prediction',...
    'Low-fidelity used observations','Mid-fidelity used observations','High-fidelity used observations')
box on 
grid on 
set(gca,'FontSize',18);

print('IMG/sc2_fx3','-dpng')
%% GreedyNAR without low-fidelity observations
[yPred, model, absId_use] = GreedyNAR(xtr,{ylf_tr,ymf_tr,yhf_tr},xte,'ard',[Ntr,Ntr/2,Ntr/4]);

figure(5)
clf
hold on 
plot(xte,ylf_te,'g-','LineWidth',2, 'MarkerSize',10)
plot(xte,ymf_te,'m-','LineWidth',2, 'MarkerSize',10)
plot(xte,yhf_te,'b-','LineWidth',2, 'MarkerSize',10)

plot(xte,yPred,'r--','LineWidth',2, 'MarkerSize',10)

% id_use = absId_use{3};
plot(xtr(absId_use{1}),ylf_tr(absId_use{1}),'g+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{2}),ymf_tr(absId_use{2}),'m+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{3}),yhf_tr(absId_use{3}),'b+','LineWidth',2, 'MarkerSize',10)
hold off

legend('Low-fidelity groundtruth','Mid-fidelity groundtruth','High-fidelity groundtruth','High-fidelity prediction',...
    'Low-fidelity used observations','Mid-fidelity used observations','High-fidelity used observations')
box on 
grid on 
set(gca,'FontSize',18);

print('IMG/greedyNAR1_fx3','-dpng')
%% GreedyNAR with low-fidelity observations
[yPred, model, absId_use] = GreedyNAR_v02(xtr,{ylf_tr,ymf_tr,yhf_tr},xte,{ylf_te,[],[]},'ard',[Ntr,Ntr/2,Ntr/4]);
% [yPred, model, absId_use] = GreedyNAR_v02(xtr,{ylf_tr,ymf_tr,yhf_tr},xte,{ylf_te,ymf_te,[]},'ard',[Ntr,Ntr/2,Ntr/4]);

figure(6)
clf
hold on 
plot(xte,ylf_te,'g-','LineWidth',2, 'MarkerSize',10)
plot(xte,ymf_te,'m-','LineWidth',2, 'MarkerSize',10)
plot(xte,yhf_te,'b-','LineWidth',2, 'MarkerSize',10)

plot(xte,yPred,'r--','LineWidth',2, 'MarkerSize',10)

% id_use = absId_use{3};
plot(xtr(absId_use{1}),ylf_tr(absId_use{1}),'g+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{2}),ymf_tr(absId_use{2}),'m+','LineWidth',2, 'MarkerSize',10)
plot(xtr(absId_use{3}),yhf_tr(absId_use{3}),'b+','LineWidth',2, 'MarkerSize',10)
hold off

legend('Low-fidelity groundtruth','Mid-fidelity groundtruth','High-fidelity groundtruth','High-fidelity prediction',...
    'Low-fidelity used observations','Mid-fidelity used observations','High-fidelity used observations')
box on 
grid on 
set(gca,'FontSize',18);
print('IMG/greedyNAR2_fx3','-dpng')