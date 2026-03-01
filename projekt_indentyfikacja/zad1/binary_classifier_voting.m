clc; clear; close all;
rng(2)
warning('off', 'all');

if isempty(gcp('nocreate'))
    parpool;
end

path_data = './Faults_database/';
errors = 1;             % errors included
fs = 20000;             % sampling rate
windowLen = 100;        % window length in samples
hop = windowLen/10;     % window shift step
tailStart = 20001;      % time of fault occurance in samples
nCases = length(errors);

S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
if numel(S0.ia) < tailStart
    error('File contains less than %d samples.', tailStart);
end
nTail = numel(S0.ia) - (tailStart-1);

ia_fault = zeros(nTail, nCases);
ib_fault = zeros(nTail, nCases);
ic_fault = zeros(nTail, nCases);
ialfa_fault = zeros(nTail, nCases);
ibeta_fault = zeros(nTail, nCases);
motor_torque_fault = zeros(nTail, nCases);

dt = 1/fs;
t_fault = (0:nTail-1) * dt;

train_plot_dir = './training_data/';
if ~exist(train_plot_dir, 'dir')
    mkdir(train_plot_dir);
end

for i = 1:nCases
    case_number = errors(i);

    hFig = figure('Visible','off');

    subplot(3,1,1)
    plot(t_fault, ia_fault(:,i),'r', ...
         t_fault, ib_fault(:,i),'g', ...
         t_fault, ic_fault(:,i),'b','LineWidth',1.1);
    grid on;
    ylabel('Currents [A]');
    title(['Fault ', num2str(case_number), ' — phase currents']);
    legend('i_a','i_b','i_c');

    subplot(3,1,2)
    plot(t_fault, ialfa_fault(:,i),'m', ...
         t_fault, ibeta_fault(:,i),'c','LineWidth',1.1);
    grid on;
    ylabel('\alpha\beta');
    title('Clarke components');
    legend('\alpha','\beta');

    subplot(3,1,3)
    plot(t_fault, motor_torque_fault(:,i),'k','LineWidth',1.2);
    grid on;
    ylabel('Torque [Nm]');
    xlabel('Time [s]');
    title('Electromagnetic torque');

    fname = fullfile(train_plot_dir, ...
        sprintf('fault%d_plot.png', case_number));
    saveas(hFig, fname);
    close(hFig);
end

fprintf('Training data plots saved to directory: %s\n', train_plot_dir);

for i = 1:nCases
    case_number = errors(i);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', case_number, '_faultType0.mat'));

    if numel(S.ia) < tailStart + nTail - 1
        error('File fault%d contains too few samples.', case_number);
    end

    ia_fault(:,i) = S.ia(tailStart:end,1);
    ib_fault(:,i) = S.ib(tailStart:end,1);
    ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1);
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1);
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

ia_correct = S.ia(1:tailStart-1,1);
ib_correct = S.ib(1:tailStart-1,1);
ic_correct = S.ic(1:tailStart-1,1);
ialfa_correct = S.ialfa(1:tailStart-1,1);
ibeta_correct = S.ibeta(1:tailStart-1,1);
motor_torque_correct = S.motor_torque(1:tailStart-1,1);

fault = [ia_fault(:), ib_fault(:), ic_fault(:), ialfa_fault(:), ibeta_fault(:), motor_torque_fault(:)];

function X = build_features_multi(mat, ~, windowLen, hop)
    % mat: N x C (C = 6: ia, ib, ic, ialfa, ibeta, torque)
    n = size(mat,1);
    C = size(mat,2);
    X = [];
    
    for s = 1:hop:(n - windowLen + 1)
        w = mat(s:s+windowLen-1, :); % windowLen x C
        f_all_channels = [];
        
        for p = 1:C
            x = w(:,p);
            
            % --- Statistical features ---
            f_stats = [rms(x), std(x), max(x), min(x), skewness(x), kurtosis(x)];
            
            % --- Spectral feature (FFT) for CURRENT channel ---
            Xf = fft(x);
            mag = abs(Xf(1:floor(windowLen/2)));
            if numel(mag) > 2
                [~, idx] = max(mag(2:end)); % Skip DC component at index 1
                idx = idx + 1;
                domAmp = mag(idx);
            else
                domAmp = 0;
            end
            
            f_all_channels = [f_all_channels, f_stats, domAmp];
        end
        
        ia = w(:,1); ib = w(:,2); ic = w(:,3);
        zero_seq = mean(ia + ib + ic);
        pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));
        
        X = [X; f_all_channels, zero_seq, pos_seq];
    end
end

X_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf];
end

mat_ok = [ia_correct, ib_correct, ic_correct, ialfa_correct, ibeta_correct, motor_torque_correct];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

Y_fault = ones(size(X_fault,1),1);
Y_ok = zeros(size(X_ok,1),1);

Xall = [X_fault; X_ok];
Yall = [Y_fault; Y_ok];
perm = randperm(size(Xall,1));
Xall = Xall(perm,:);
Yall = Yall(perm);

cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:);
Ytrain = Yall(training(cv));
Xtest = Xall(test(cv),:);
Ytest = Yall(test(cv));

mu = mean(Xtrain,1);
sigma = std(Xtrain,[],1);
sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma;
Xtest_norm = (Xtest - mu) ./ sigma;

inputSize = size(Xtrain_norm,2);

layers = [
    featureInputLayer(inputSize)

    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(32)
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];


options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',1e-3, ...
    'Verbose',true);

Ytrain_cat = categorical(Ytrain);
dsTrain = arrayDatastore(Xtrain_norm,'IterationDimension',1);
trainTbl = combine(dsTrain, arrayDatastore(Ytrain_cat));

net = trainNetwork(Xtrain_norm, Ytrain_cat, layers, options);

Ypred_probs = predict(net, Xtest_norm);

prob_plot_dir = './training_data/probability/';
if ~exist(prob_plot_dir, 'dir')
    mkdir(prob_plot_dir);
end

[~, idxmax] = max(Ypred_probs, [], 2);
Ypred = idxmax - 1; % klasy 0/1

confMat = confusionmat(Ytest, Ypred);

matrix_dir = './matrix/';
if ~exist(matrix_dir, 'dir')
    mkdir(matrix_dir);
end

case_number = errors(1);

hFig = figure('Visible','off');
confusionchart(Ytest, Ypred, ...
    'Title', sprintf('Confusion matrix for error %d', case_number), ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

fname = fullfile(matrix_dir, ...
    sprintf('confusion_matrix_%d.png', case_number));

saveas(hFig, fname);
close(hFig);

fprintf('Confusion matrix saved to file: %s\n', fname);

figure;
confusionchart(Ytest, Ypred, ...
    'Title', sprintf('Confusion matrix for error %d', case_number), ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

acc = sum(Ypred == Ytest) / numel(Ytest);
fprintf('Accuracy = %.2f%%\n', acc * 100);

scoresPos = Ypred_probs(:, 2);
[Xroc, Yroc, ~, AUC] = perfcurve(Ytest, scoresPos, 1);
fprintf('AUC = %.3f\n', AUC);

plot_dir = './plots_binary/';
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

files = dir(fullfile(path_data, '*_faultType0.mat'));
windowLen_samples = windowLen; 
dt = 1/fs; 

mu_val = mu;
sigma_val = sigma;

net_gpu = net; 

fprintf('================================== \n')
fprintf('Starting parallel bulk plot generation...\n')

all_files_latencies = nan(length(files), 1);

parfor f = 1:length(files)
    filename = files(f).name;
    case_label = erase(filename, '_faultType0.mat');

    S_curr = load(fullfile(path_data, filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];

    nSamples = size(mat_signal,1);
    nChannels = size(mat_signal,2);
    Ypred_all = zeros(nSamples,1);

    found_fault_local = false;
    det_time_local = NaN;

    for s = 1:nSamples
        window = zeros(windowLen_samples, nChannels);
        if s < windowLen_samples
            window(end-s+1:end, :) = mat_signal(1:s, :);
        else
            window(:, :) = mat_signal(s-windowLen_samples+1:s, :);
        end

        Xwin = build_features_multi(window, fs, windowLen_samples, windowLen_samples);
        Xwin_norm = (Xwin - mu_val) ./ sigma_val;

        Yprob = predict(net_gpu, Xwin_norm, 'ExecutionEnvironment', 'auto');
        [~, idxmax] = max(Yprob,[],2);
        Ypred_all(s) = idxmax - 1; 
        
        if ~found_fault_local && Ypred_all(s) == 1 && s >= tailStart
            det_time_local = (s - tailStart) * dt;
            found_fault_local = true;
        end

        all_files_latencies(f) = det_time_local;
    end

    t_plot = (0:nSamples-1) * dt;
    torque_plot = mat_signal(:,6);

    hFig = figure('Visible', 'off'); 
    yyaxis left
    plot(t_plot, torque_plot, 'b', 'LineWidth', 1.2);
    ylabel('Torque [Nm]');
    grid on;

    yyaxis right
    stairs(t_plot, Ypred_all, 'r', 'LineWidth', 1.5);
    ylabel('Classification (0 / 1)');
    ylim([-0.1 1.1]);

    xlabel('Time [s]');
    title(['Detection Result: ', case_label]);

    saveas(hFig, fullfile(plot_dir, [case_label, '_plot.png']));
    close(hFig); 

    if found_fault_local
        fprintf('File: %s | Detection Time: %.2f ms\n', filename, det_time_local * 1000);
    else
        fprintf('File: %s | NO FAULT DETECTED\n', filename);
    end

    fprintf('Processed: %s\n', case_label);

end

mat_signal = [S.ia, S.ib, S.ic, S.ialfa, S.ibeta, S.motor_torque];
nSamples = size(mat_signal,1);
nChannels = size(mat_signal,2);

Ypred_all = zeros(nSamples,1);   % 0 – brak błędu, 1 – błąd

found_fault = false;
first_detection_time = NaN;
fault_start_idx = tailStart;

voteLen = 10;                     % voting window length
voteBuffer = zeros(voteLen,1);
Ypred_vote = zeros(nSamples,1);   % voting results

found_fault_vote = false;
first_detection_time_vote = NaN;

for s = 1:nSamples

    window = zeros(windowLen_samples, nChannels);
    if s < windowLen_samples
        window(end-s+1:end, :) = mat_signal(1:s, :);
    else
        window(:, :) = mat_signal(s-windowLen_samples+1:s, :);
    end

    Xwin = build_features_multi(window, fs, windowLen_samples, windowLen_samples);
    Xwin_norm = (Xwin - mu) ./ sigma;

    Yprob = predict(net, Xwin_norm);
    [~, idxmax] = max(Yprob,[],2);
    Ypred = idxmax - 1;
    Ypred_all(s) = Ypred;

    voteBuffer = [voteBuffer(2:end); Ypred];

    nVotes = sum(voteBuffer);

    if nVotes >= ceil(voteLen/2)
        Ypred_vote(s) = 1;
    else
        Ypred_vote(s) = 0;
    end

    if ~found_fault_vote && Ypred_vote(s) == 1 && s >= fault_start_idx
        first_detection_time_vote = (s - fault_start_idx) * dt;
        found_fault_vote = true;
    end

    if ~found_fault && Ypred == 1 && s >= fault_start_idx
        first_detection_time = (s - fault_start_idx) * dt;
        found_fault = true;
    end

end

if ~found_fault
    fprintf('Error not detected using ORIGINAL method.\n');
end
if ~found_fault_vote
    fprintf('Error not detected using VOTING method.\n');
end

t = (0:nSamples-1) * dt;
torque = mat_signal(:,6);

figure;
subplot(2,1,1);
yyaxis left
plot(t, torque, 'b', 'LineWidth', 1.2);
ylabel('Torque [Nm]'); grid on;

yyaxis right
stairs(t, Ypred_all, 'r', 'LineWidth', 1.5);
ylabel('Detection (0 / 1)');
ylim([-0.1 1.1]);
title('Original error detection');
xlabel('Time [s]');
legend('Torque','ORIGINAL detection','Location','best');

subplot(2,1,2);
yyaxis left
plot(t, torque, 'b', 'LineWidth', 1.2);
ylabel('Torque [Nm]'); grid on;

yyaxis right
stairs(t, Ypred_vote, 'm', 'LineWidth', 1.5);
ylabel('Detection (0 / 1)');
ylim([-0.1 1.1]);
title('Error detection using VOTING');
xlabel('Time [s]');
legend('Torque','Detection using VOTING','Location','best');

fprintf('\n=================== Detection times ===================\n');

if found_fault
    fprintf('First detection (original):      %.2f ms\n', first_detection_time * 1000);
else
    fprintf('No detection using ORIGINAL method.\n');
end

if found_fault_vote
    fprintf('First detection (voting):      %.2f ms\n', first_detection_time_vote * 1000);
else
    fprintf('No detection using VOTING.\n');
end

fprintf('=====================================================\n');
