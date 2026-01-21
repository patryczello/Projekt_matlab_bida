clc; clear; close all;
%% Ustawienia użytkownika
rng(2) 
path_data = './Faults_database/';
errors = zeros(1, 64);
for i = 0:63
    errors(i+1) = i;
end       
fs = 20000;          
windowLen = 1000;      
hop = windowLen/10;     
tailStart = 20001;   
nCases = length(errors);

if isempty(gcp('nocreate'))
    parpool; 
end

%% Wczytaj i przygotuj dane (Logic unchanged)
S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
nTail = numel(S0.ia) - (tailStart-1);

ia_fault = zeros(nTail, nCases); ib_fault = zeros(nTail, nCases); ic_fault = zeros(nTail, nCases);
ialfa_fault = zeros(nTail, nCases); ibeta_fault = zeros(nTail, nCases); motor_torque_fault = zeros(nTail, nCases);

for i = 1:nCases
    S = load(sprintf('%s%s%d%s', path_data, 'fault', errors(i), '_faultType0.mat'));
    ia_fault(:,i) = S.ia(tailStart:end,1); ib_fault(:,i) = S.ib(tailStart:end,1); ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1); ibeta_fault(:,i) = S.ibeta(tailStart:end,1); motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
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

%% Przygotowanie i Trening Sieci
X_fault = []; Y_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf]; Y_fault = [Y_fault; repmat(errors(i), size(Xf,1), 1)];
end
mat_ok = [ia_poprawne, ib_poprawne, ic_poprawne, ialfa_poprawne, ibeta_poprawne, motor_torque_poprawne];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop); Y_ok = zeros(size(X_ok,1),1);

Xall = [X_fault; X_ok]; Yall = [Y_fault; Y_ok]; perm = randperm(size(Xall,1));
Xall = Xall(perm,:); Yall = Yall(perm);
cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:); Ytrain = Yall(training(cv));
Xtest = Xall(test(cv),:); Ytest = Yall(test(cv));

mu = mean(Xtrain,1); sigma = std(Xtrain,[],1); sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma; Xtest_norm = (Xtest - mu) ./ sigma;

inputSize = size(Xtrain_norm,2); numClasses = numel(unique(Yall));
layers = [featureInputLayer(inputSize), ...
          fullyConnectedLayer(256), ...
          batchNormalizationLayer, ...
          reluLayer, ...
          fullyConnectedLayer(128), ...
          batchNormalizationLayer, ...
          reluLayer, ...
          dropoutLayer(0.5), ...
          fullyConnectedLayer(64), ...
          reluLayer, ...
          fullyConnectedLayer(numClasses), ...
          softmaxLayer, ...
          classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 256, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 40, ...
    'Verbose', true, ...
    'Plots', 'none');
Ytrain_cat = categorical(Ytrain);
net = trainNetwork(Xtrain_norm, Ytrain_cat, layers, options);

%% Ocena Wyników
Ypred_probs = predict(net, Xtest_norm);
cats = categories(Ytrain_cat);           
[~, idxmax] = max(Ypred_probs,[],2);     
Ypred = str2double(cats(idxmax));        
fprintf('Accuracy = %.2f%%\n', (sum(Ypred==Ytest)/numel(Ytest))*100);
figure; confusionchart(Ytest, Ypred, 'Title', 'Multi-Class Fault Confusion Matrix');

%% Bulk Processing - Strict Online Simulation with Printed Info
plot_dir = './plots_multi/';
if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
files = dir(fullfile(path_data, '*_faultType0.mat'));

% === Dynamiczne mapowanie klas (0..63) ===
errors = 1:63;

error_vals  = [0, errors];                 % [0 1 2 ... 63]
error_names = "Fault " + string(error_vals);
error_names(1) = "Healthy";                % klasa 0

dt = 1/fs; 
mu_val = mu; 
sigma_val = sigma; 
net_gpu = net; 

all_files_latencies = nan(length(files), 1); 

fprintf('================================== \n')
fprintf('Starting parallel bulk plot generation (Multi-class Online)...\n')

parfor f = 1:length(files)

    filename = files(f).name;

    % --- Determine True Label for Title ---
    true_desc = "Healthy"; 
    for e_idx = 1:numel(errors)
        tag = sprintf('fault%d_', errors(e_idx));
        if contains(filename, tag)
            true_desc = "Fault " + errors(e_idx);
            break;
        end
    end

    case_label = erase(filename, '_faultType0.mat');
    S_curr = load(fullfile(path_data, filename));

    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, ...
                  S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];

    nSamples = size(mat_signal,1);

    pred_over_time   = zeros(nSamples,1);
    found_fault_local = false;
    det_time_local    = NaN;

    % --- Online Simulation Loop ---
    for s = windowLen:nSamples

        % 1. Extract Buffer (Online sliding window)
        window = mat_signal(s-windowLen+1:s, :);

        % 2. Feature Extraction & Prediction
        % hop = windowLen -> 1 wektor cech
        Xwin = build_features_multi(window, fs, windowLen, windowLen);
        Xwin_norm = (Xwin - mu_val) ./ sigma_val;

        Yprob = predict(net_gpu, Xwin_norm, 'ExecutionEnvironment', 'auto');
        [~, idx] = max(Yprob, [], 2);

        % --- BEZPIECZNE MAPOWANIE: indeks klasy → etykieta liczbowa ---
        if idx < 1
            idx = 1;
        elseif idx > numel(error_vals)
            warning('Predicted class index %d exceeds known classes. Clipping.', idx);
            idx = numel(error_vals);
        end

        current_pred = error_vals(idx);
        pred_over_time(s) = current_pred;

        % 3. Real-time Detection Logic
        if ~found_fault_local && current_pred > 0 && s >= tailStart
            det_time_local = (s - tailStart) * dt;
            found_fault_local = true;
        end
    end

    all_files_latencies(f) = det_time_local;

    % --- Generate Plot ---
    t_plot = (0:nSamples-1) * dt;
    hFig = figure('Visible', 'off'); 

    yyaxis left
    plot(t_plot, mat_signal(:,6), 'b'); 
    ylabel('Torque [Nm]');
    grid on;

    yyaxis right
    stairs(t_plot, pred_over_time, 'r', 'LineWidth', 1.2); 
    ylabel('Classification');

    yticks(error_vals);
    yticklabels(error_names);
    ylim([-1 max(errors) + 5]); 

    xlabel('Time [s]');
    title({['File: ', strrep(case_label, '_', '\_')], ...
           ['Actual Status: ', char(true_desc)]});

    % Add indicator for fault injection point
    xline(tailStart * dt, '--k', 'Fault Start');

    saveas(hFig, fullfile(plot_dir, ...
           strrep(filename, '.mat', '_multi_plot.png')));

    % --- Printed Info ---
    if found_fault_local
        fprintf('File: %s | True: %s | Det. Time: %.2f ms\n', ...
                filename, true_desc, det_time_local * 1000);
    else
        fprintf('File: %s | True: %s | NO FAULT DETECTED\n', ...
                filename, true_desc);
    end

    close(hFig);
end


%% Statistical Report (Restored)
valid_latencies = all_files_latencies(~isnan(all_files_latencies)) * 1000; 
detection_rate = (sum(~isnan(all_files_latencies)) / length(files)) * 100;
fprintf('\n================ MULTI-CLASS STATISTICAL REPORT ================\n');
fprintf('Total Files Processed: %d\n', length(files));
fprintf('Detection Rate: %.2f%%\n', detection_rate);
if ~isempty(valid_latencies)
    fprintf('Mean Detection Time: %.2f ms\n', mean(valid_latencies));
    fprintf('Std Deviation: %.2f ms\n', std(valid_latencies));
    figure; histogram(valid_latencies, 15, 'FaceColor', '#D95319'); 
    xlabel('Latency [ms]'); ylabel('Count');
    title('Detection Latency Distribution');
end