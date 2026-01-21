clc; clear; close all;
%% Ustawienia użytkownika
rng(2) 
path_data = './Faults_database/';
errors = 1:63;
fs = 20000;          
windowLen = 2000;      
hop = windowLen/10;
tailStart = 20001;   
nCases = length(errors);

if isempty(gcp('nocreate')), parpool; end

%% 1. Wczytaj i przygotuj dane (Faults Only)
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

%% 2. Funkcja budująca cechy (No changes to logic, just usage)
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

%% 3. Przygotowanie i Trening Sieci
Xall = []; Yall = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    Xall = [Xall; Xf]; 
    Yall = [Yall; repmat(errors(i), size(Xf,1), 1)];
end

% STRICT MAPPING: Force 1:63 categorical levels
Yall_cat = categorical(Yall, 1:63); 

perm = randperm(size(Xall,1));
Xall = Xall(perm,:); Yall_cat = Yall_cat(perm);

cv = cvpartition(Yall_cat,'HoldOut',0.3);
Xtrain = Xall(training(cv),:); Ytrain = Yall_cat(training(cv));
Xtest = Xall(test(cv),:); Ytest = Yall_cat(test(cv));

mu = mean(Xtrain,1); sigma = std(Xtrain,[],1); sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma; Xtest_norm = (Xtest - mu) ./ sigma;

inputSize = size(Xtrain_norm,2); 
numClasses = 63;

layers = [featureInputLayer(inputSize)
          fullyConnectedLayer(512)
          batchNormalizationLayer
          reluLayer
          dropoutLayer(0.3)
          fullyConnectedLayer(256)
          batchNormalizationLayer
          reluLayer
          fullyConnectedLayer(numClasses)
          softmaxLayer
          classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-2, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none');

net = trainNetwork(Xtrain_norm, Ytrain, layers, options);

%% 4. Ocena Wyników
Ypred = classify(net, Xtest_norm);
fprintf('Accuracy = %.2f%%\n', (sum(Ypred == Ytest)/numel(Ytest))*100);
figure; confusionchart(Ytest, Ypred, 'Title', '63-Class Fault Matrix');

%% 5. Bulk Processing - Online Simulation
plot_dir = './plots_multi/';
if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
files = dir(fullfile(path_data, '*_faultType0.mat'));

% Extract category list for safe mapping inside parfor
net_cats = categories(Ytrain); 
mu_v = mu; sigma_v = sigma;

parfor f = 1:length(files)
    filename = files(f).name;
    % Detect true label from filename
    true_desc = "Unknown";
    for e_val = 1:63
        if contains(filename, sprintf('fault%d_', e_val))
            true_desc = "Fault " + e_val; break;
        end
    end
    
    S_curr = load(fullfile(path_data, filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];
    nSamples = size(mat_signal,1);
    pred_over_time = zeros(nSamples,1);
    
    % Online Loop
    for s = windowLen:100:nSamples % Step by 100 to speed up simulation
        window = mat_signal(s-windowLen+1:s, :);
        Xf = build_features_multi(window, fs, windowLen, windowLen);
        X_norm = (Xf - mu_v) ./ sigma_v;
        
        [Yp, scores] = classify(net, X_norm);
        % Map the categorical back to numeric
        pred_over_time(s-99:s) = str2double(char(Yp));
    end
    
    % --- Plotting ---
    t = (0:nSamples-1)/fs;
    h = figure('Visible', 'off');
    yyaxis left; plot(t, mat_signal(:,6)); ylabel('Torque');
    yyaxis right; stairs(t, pred_over_time, 'r'); ylabel('Class (1-63)');
    ylim([0 65]); title(sprintf('File: %s | True: %s', filename, true_desc));
    xline(tailStart/fs, '--k', 'Fault Start');
    saveas(h, fullfile(plot_dir, [filename '.png']));
    close(h);
end