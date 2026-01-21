clc; clear; close all;

%% Ustawienia użytkownika
rng(2) 
path_data = './Faults_database/';
errors = 0:63;       
fs = 20000;          
windowLen = 100;     
hop = windowLen/10;  
tailStart = 20001;   
nCases = length(errors);

if isempty(gcp('nocreate'))
    parpool; 
end

%% Wczytaj i przygotuj dane
S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
nTail = numel(S0.ia) - (tailStart-1);

ia_fault = zeros(nTail, nCases); ib_fault = zeros(nTail, nCases); ic_fault = zeros(nTail, nCases);
ialfa_fault = zeros(nTail, nCases); ibeta_fault = zeros(nTail, nCases); motor_torque_fault = zeros(nTail, nCases);

for i = 1:nCases
    S = load(sprintf('%s%s%d%s', path_data, 'fault', errors(i), '_faultType0.mat'));
    ia_fault(:,i) = S.ia(tailStart:end,1); 
    ib_fault(:,i) = S.ib(tailStart:end,1); 
    ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1); 
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1); 
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

ia_poprawne = S.ia(1:tailStart-1,1); ib_poprawne = S.ib(1:tailStart-1,1); ic_poprawne = S.ic(1:tailStart-1,1);
ialfa_poprawne = S.ialfa(1:tailStart-1,1); ibeta_poprawne = S.ibeta(1:tailStart-1,1); motor_torque_poprawne = S.motor_torque(1:tailStart-1,1);

%% Funkcja budująca cechy (Multichannel)
function X = build_features_multi(mat, ~, windowLen, hop)
    n = size(mat,1); C = size(mat,2); X = [];
    for s = 1:hop:(n - windowLen + 1)
        w = mat(s:s+windowLen-1, :); f_phase = [];
        for p = 1:C
            x = w(:,p);
            f_phase = [f_phase, rms(x), std(x), max(x), min(x), skewness(x), kurtosis(x)];
        end
        ia = w(:,1); ib = w(:,2); ic = w(:,3);
        zero_seq = mean(ia + ib + ic);
        pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));
        torque_mean = mean(w(:,6)); torque_std = std(w(:,6));
        Xf = fft(ia); mag = abs(Xf(1:floor(windowLen/2)));
        if numel(mag) > 2, [~, idx] = max(mag(2:end)); idx = idx+1; domAmp = mag(idx); else, domAmp = 0; end
        X = [X; f_phase, zero_seq, pos_seq, torque_mean, torque_std, domAmp];
    end
end

%% Przygotowanie danych i trening sieci
X_fault = []; Y_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf]; 
    Y_fault = [Y_fault; repmat(errors(i), size(Xf,1), 1)];
end

mat_ok = [ia_poprawne, ib_poprawne, ic_poprawne, ialfa_poprawne, ibeta_poprawne, motor_torque_poprawne];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop); 
Y_ok = zeros(size(X_ok,1),1);

Xall = [X_fault; X_ok]; 
Yall = [Y_fault; Y_ok]; 
perm = randperm(size(Xall,1));
Xall = Xall(perm,:); 
Yall = Yall(perm);

cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:); Ytrain = Yall(training(cv));
Xtest = Xall(test(cv),:); Ytest = Yall(test(cv));

mu = mean(Xtrain,1); sigma = std(Xtrain,[],1); sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma; 
Xtest_norm = (Xtest - mu) ./ sigma;

inputSize = size(Xtrain_norm,2); 
numClasses = numel(unique(Yall));
layers = [featureInputLayer(inputSize), ...
          fullyConnectedLayer(256), batchNormalizationLayer, reluLayer, ...
          fullyConnectedLayer(128), batchNormalizationLayer, reluLayer, ...
          dropoutLayer(0.5), fullyConnectedLayer(64), reluLayer, ...
          fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, 'MiniBatchSize', 128, ...
    'LearnRateSchedule','piecewise', 'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 20, 'Verbose', true, 'Plots', 'none');

Ytrain_cat = categorical(Ytrain);
net = trainNetwork(Xtrain_norm, Ytrain_cat, layers, options);

%% Bulk Processing - Online Simulation + Confusion Matrices
plot_dir = './plots_multi/'; if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
cm_dir = './confusion_matrices/'; if ~exist(cm_dir, 'dir'), mkdir(cm_dir); end

files = dir(fullfile(path_data, '*_faultType0.mat'));
error_vals = [0, errors]; 
error_names = ["Healthy", "Fault 1", "Fault 2", "Fault 4", "Fault 8", "Fault 16", "Fault 32"];
dt = 1/fs; mu_val = mu; sigma_val = sigma; net_gpu = net;

for f = 1:length(files)
    filename = files(f).name;
    case_label = erase(filename, '_faultType0.mat');
    
    S_curr = load(fullfile(path_data, filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];
    nSamples = size(mat_signal,1);
    
    pred_over_time = zeros(nSamples,1);
    
    % Określenie prawdziwej etykiety
    true_label = 0;
    for e_idx = 1:length(errors)
        if contains(filename, sprintf('fault%d_', errors(e_idx)))
            true_label = errors(e_idx);
            break;
        end
    end
    Ytrue_over_time = true_label * ones(nSamples,1);
    
    % Online Sliding Window
    for s = windowLen:nSamples
        window = mat_signal(s-windowLen+1:s, :);
        Xwin = build_features_multi(window, fs, windowLen, windowLen);
        Xwin_norm = (Xwin - mu_val) ./ sigma_val;
        Yprob = predict(net_gpu, Xwin_norm, 'ExecutionEnvironment', 'auto');
        [~, idx] = max(Yprob,[],2);
        pred_over_time(s) = str2double(categories(Ytrain_cat)(idx));
    end
    
    % --- Confusion Matrix per File ---
    Ytrue_cat = categorical(Ytrue_over_time, error_vals, error_names);
    Ypred_cat = categorical(pred_over_time, error_vals, error_names);
    hFig = figure('Visible','off');
    cm = confusionchart(Ytrue_cat, Ypred_cat, 'Title',['Confusion Matrix: ', case_label], ...
                        'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
    saveas(hFig, fullfile(cm_dir, [case_label, '_confusion_matrix.png']));
    close(hFig);
    
    fprintf('Processed file: %s | True Label: %d\n', filename, true_label);
end

fprintf('All confusion matrices saved in folder: %s\n', cm_dir);
