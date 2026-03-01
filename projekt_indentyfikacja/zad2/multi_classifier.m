clc; clear; close all;

rng(2) 
path_data = './Faults_database/';

fault_indices = 1:63;      % errors included
fs = 20000;                % sampling rate
windowLen = 1000;          % window length in samples
hop = windowLen/10;        % window shift step
tailStart = 20001;         % time of fault occurance in samples
nFaultCases = length(fault_indices);

first_file = dir(fullfile(path_data, 'fault1_faultType0.mat'));
S_ref = load(fullfile(path_data, first_file(1).name));
nTail = numel(S_ref.ia) - (tailStart-1);

ia_fault = zeros(nTail, nFaultCases); ib_fault = zeros(nTail, nFaultCases); 
ic_fault = zeros(nTail, nFaultCases); ialfa_fault = zeros(nTail, nFaultCases); 
ibeta_fault = zeros(nTail, nFaultCases); motor_torque_fault = zeros(nTail, nFaultCases);

fprintf('Loading %d Fault cases...\n', nFaultCases);
for i = 1:nFaultCases
    errVal = fault_indices(i);
    target_file = sprintf('%sfault%d_faultType0.mat', path_data, errVal);
    
    if exist(target_file, 'file')
        S = load(target_file);
        ia_fault(:,i) = S.ia(tailStart:end,1); 
        ib_fault(:,i) = S.ib(tailStart:end,1); 
        ic_fault(:,i) = S.ic(tailStart:end,1);
        ialfa_fault(:,i) = S.ialfa(tailStart:end,1); 
        ibeta_fault(:,i) = S.ibeta(tailStart:end,1); 
        motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
    end
end

function X = build_features_multi(mat, ~, windowLen, hop)
    n = size(mat,1); C = size(mat,2); X = [];

    for s = 1:hop:(n - windowLen + 1)
        w = mat(s:s+windowLen-1, :); f_phase = [];

        for p = 1:C
            x = w(:,p);
            curr_rms = rms(x);
            p2p = max(x) - min(x);
            crest = max(abs(x)) / (curr_rms + 1e-6);
            f_phase = [f_phase, curr_rms, std(x), skewness(x), kurtosis(x), p2p, crest];
        end

        ia = w(:,1); ib = w(:,2); ic = w(:,3);
        zero_seq = mean(ia + ib + ic);
        pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));
        torque_mean = mean(w(:,6)); torque_std = std(w(:,6));
        Xf = fft(ia); mag = abs(Xf(1:floor(windowLen/2)));

        if numel(mag) > 2
            [maxAmp, ~] = max(mag(2:end)); 
            thd_val = sqrt(sum(mag.^2) - maxAmp^2) / (maxAmp + 1e-6);
            domAmp = maxAmp;
        else
            domAmp = 0; thd_val = 0; 
        end

        X = [X; f_phase, zero_seq, pos_seq, torque_mean, torque_std, domAmp, thd_val];
    end
end

Xall = []; Yall = [];
for i = 1:nFaultCases
    if any(ia_fault(:,i))
        mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ...
               ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
        Xf = build_features_multi(mat, fs, windowLen, hop);
        Xall = [Xall; Xf]; 
        Yall = [Yall; repmat(fault_indices(i), size(Xf,1), 1)];
    end

end

Yall_cat = categorical(Yall);
numClasses = numel(categories(Yall_cat));

perm = randperm(size(Xall,1));
Xall = Xall(perm,:); Yall_cat = Yall_cat(perm);
cv = cvpartition(Yall_cat,'HoldOut',0.3);
Xtrain = Xall(training(cv),:); Ytrain = Yall_cat(training(cv));
Xtest = Xall(test(cv),:); Ytest = Yall_cat(test(cv));

mu = mean(Xtrain,1); sigma = std(Xtrain,[],1); sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma; Xtest_norm = (Xtest - mu) ./ sigma;

inputSize = size(Xtrain_norm,2); 
numClasses = numel(categories(Yall_cat));
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 75, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none');

net = trainNetwork(Xtrain_norm, Ytrain, layers, options);
Ypred = classify(net, Xtest_norm);
accuracy = (sum(Ypred == Ytest)/numel(Ytest))*100;

fprintf('Final Test Accuracy = %.2f%%\n', accuracy);
figure; confusionchart(Ytest, Ypred, 'Title', 'Fault Diagnosis');

% Online simulation
plot_dir = './plots_multi/';

if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
files = dir(fullfile(path_data, '*_faultType0.mat'));
dt = 1/fs; 
all_files_latencies = nan(length(files), 1); 

classNames = double(string(categories(Yall_cat)));
fprintf('\nStarting Online Simulation and Plotting...\n');
parfor f = 1:length(files)
    filename = files(f).name;
    
    true_label_val = 0;
    for e_val = 1:63
        if contains(filename, sprintf('fault%d_', e_val))
            true_label_val = e_val; break;
        end
    end
    
    S_curr = load(fullfile(path_data, filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];
    nSamples = size(mat_signal,1);
    pred_over_time = zeros(nSamples,1);
    found_fault_local = false;
    det_time_local = NaN;

    for s = windowLen:100:nSamples 
        window = mat_signal(s-windowLen+1:s, :);
        Xwin = build_features_multi(window, fs, windowLen, windowLen);
        Xwin_norm = (Xwin - mu) ./ sigma;
        
        [Ypred_val, ~] = classify(net, Xwin_norm);
        current_pred = double(string(Ypred_val));
        
        pred_over_time(s-99:s) = current_pred;
        
        if ~found_fault_local && current_pred > 0 && s >= tailStart
            det_time_local = (s - tailStart) * dt;
            found_fault_local = true;
        end
    end

    all_files_latencies(f) = det_time_local;

    t_plot = (0:nSamples-1) * dt;
    hFig = figure('Visible', 'off');
    yyaxis left

    plot(t_plot, mat_signal(:,6), 'b'); ylabel('Torque [Nm]'); grid on;
    yyaxis right

    stairs(t_plot, pred_over_time, 'r', 'LineWidth', 1.2); ylabel('Predicted Class');
    ylim([-1 65]);
    xlabel('Time [s]');
    title_str = sprintf('File: %s | True Label: %d', strrep(filename, '_', '\_'), true_label_val);
    title({title_str, ['Detection Latency: ', num2str(det_time_local, '%.4f'), ' s']});
    xline(tailStart * dt, '--k', 'Fault Injection');
    saveas(hFig, fullfile(plot_dir, [filename '.png']));
    close(hFig);
end
