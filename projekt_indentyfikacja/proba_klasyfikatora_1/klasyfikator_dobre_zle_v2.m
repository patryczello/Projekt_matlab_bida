clc; clear; close all;

%% ===== Ustawienia użytkownika =====
rng(2)
warning('off','all');

if isempty(gcp('nocreate'))
    parpool; % starts parallel pool
end

path_data = './Faults_database/';
errors = 1:5;      % wszystkie przypadki błędów
fs = 20000;         % częstotliwość próbkowania
windowLen = 100;    % długość okna (próbki)
hop = windowLen/10; % hop
tailStart = 20001;  % początek części "fault"
nCases = length(errors);

%% ===== Wczytanie pierwszego pliku do ustalenia nTail =====
S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
if numel(S0.ia) < tailStart
    error('Plik ma mniej niż %d próbek.', tailStart);
end
nTail = numel(S0.ia) - (tailStart-1);

%% ===== Prealokacja macierzy =====
ia_fault = zeros(nTail,nCases);
ib_fault = zeros(nTail,nCases);
ic_fault = zeros(nTail,nCases);
ialfa_fault = zeros(nTail,nCases);
ibeta_fault = zeros(nTail,nCases);
motor_torque_fault = zeros(nTail,nCases);

%% ===== Wczytaj wszystkie przypadki fault =====
for i = 1:nCases
    case_number = errors(i);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', case_number, '_faultType0.mat'));

    ia_fault(:,i) = S.ia(tailStart:end,1);
    ib_fault(:,i) = S.ib(tailStart:end,1);
    ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1);
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1);
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

%% ===== OK signals =====
Slast = S; % ostatni wczytany plik jako przykład
ia_ok = Slast.ia(1:tailStart-1,1);
ib_ok = Slast.ib(1:tailStart-1,1);
ic_ok = Slast.ic(1:tailStart-1,1);
ialfa_ok = Slast.ialfa(1:tailStart-1,1);
ibeta_ok = Slast.ibeta(1:tailStart-1,1);
motor_torque_ok = Slast.motor_torque(1:tailStart-1,1);

%% ===== Oś czasu =====
dt = 1/fs;
t_fault = (0:nTail-1)*dt;

%% ===== Folder na training plots =====
train_plot_dir = './training_data/';
if ~exist(train_plot_dir,'dir')
    mkdir(train_plot_dir);
end

%% ===== Wykresy danych treningowych dla wszystkich przypadków =====
for i = 1:nCases
    hFig = figure('Visible','off');
    subplot(3,1,1)
    plot(t_fault, ia_fault(:,i),'r', t_fault, ib_fault(:,i),'g', t_fault, ic_fault(:,i),'b','LineWidth',1.1)
    grid on; ylabel('Prądy [A]'); title(['Fault ', num2str(errors(i)),' — prądy fazowe']); legend('i_a','i_b','i_c');

    subplot(3,1,2)
    plot(t_fault, ialfa_fault(:,i),'m', t_fault, ibeta_fault(:,i),'c','LineWidth',1.1)
    grid on; ylabel('\alpha\beta'); title('Składowe Clarke'); legend('\alpha','\beta');

    subplot(3,1,3)
    plot(t_fault, motor_torque_fault(:,i),'k','LineWidth',1.2)
    grid on; ylabel('Moment [Nm]'); xlabel('Czas [s]'); title('Moment elektromagnetyczny');

    fname = fullfile(train_plot_dir, sprintf('fault%d_plot.png', errors(i)));
    saveas(hFig,fname); close(hFig);
end

fprintf('Zapisano wszystkie training data do folderu: %s\n', train_plot_dir);

%% ===== Budowa cech (funkcja w osobnym pliku build_features_multi.m) =====
% mat: N x C (C=6)
% Zwraca NxM macierz cech
% Funkcja: rms, std, max, min, skewness, kurtosis + FFT max + zero_seq + pos_seq

%% ===== Przygotowanie danych ML =====
X_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf];
end

mat_ok = [ia_ok, ib_ok, ic_ok, ialfa_ok, ibeta_ok, motor_torque_ok];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

Y_fault = ones(size(X_fault,1),1);
Y_ok = zeros(size(X_ok,1),1);

Xall = [X_fault; X_ok];
Yall = [Y_fault; Y_ok];
perm = randperm(size(Xall,1));
Xall = Xall(perm,:);
Yall = Yall(perm);

%% ===== Podział train/test =====
cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:);
Ytrain = Yall(training(cv));
Xtest = Xall(test(cv),:);
Ytest = Yall(test(cv));

%% ===== Normalizacja =====
mu = mean(Xtrain,1); sigma = std(Xtrain,[],1); sigma(sigma==0)=1;
Xtrain_norm = (Xtrain - mu) ./ sigma;
Xtest_norm = (Xtest - mu) ./ sigma;

%% ===== Sieć neuronowa =====
inputSize = size(Xtrain_norm,2);
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(128); batchNormalizationLayer; reluLayer
    fullyConnectedLayer(64); batchNormalizationLayer; reluLayer; dropoutLayer(0.5)
    fullyConnectedLayer(32); reluLayer
    fullyConnectedLayer(2); softmaxLayer; classificationLayer
];
options = trainingOptions('adam','MaxEpochs',20,'MiniBatchSize',128,'InitialLearnRate',1e-3,'Verbose',true);
Ytrain_cat = categorical(Ytrain);
net = trainNetwork(Xtrain_norm,Ytrain_cat,layers,options);

%% ===== Ocena i macierz pomyłek dla wszystkich przypadków =====
Ypred_probs = predict(net,Xtest_norm);
[~, idxmax] = max(Ypred_probs,[],2);
Ypred = idxmax-1;

matrix_dir = './matrix/';
if ~exist(matrix_dir,'dir'); mkdir(matrix_dir); end

hFig = figure('Visible','off');
confusionchart(Ytest,Ypred,'Title','Macierz pomyłek zbiorcza','ColumnSummary','column-normalized','RowSummary','row-normalized');
saveas(hFig, fullfile(matrix_dir,'confusion_matrix_all.png'));
close(hFig);

acc = sum(Ypred==Ytest)/numel(Ytest);
fprintf('Zbiorcza accuracy = %.2f%%\n', acc*100);

scoresPos = Ypred_probs(:,2);
[~,~,~,AUC] = perfcurve(Ytest,scoresPos,1);
fprintf('Zbiorcza AUC = %.3f\n', AUC);

%% ===== Bulk processing: predykcja + głosowanie + zapis wykresów + macierz pomyłek =====
plot_dir = './plots_binary/';
matrix_dir = './matrix/';
if ~exist(plot_dir,'dir'); mkdir(plot_dir); end
if ~exist(matrix_dir,'dir'); mkdir(matrix_dir); end
voteLen = 200;

files = dir(fullfile(path_data,'*_faultType0.mat'));
for f = 1:length(files)
    filename = files(f).name;
    case_label = erase(filename,'_faultType0.mat');

    S_curr = load(fullfile(path_data,filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];
    nSamples = size(mat_signal,1);

    % ---- Predykcja + głosowanie ----
    voteBuffer = zeros(voteLen,1);
    Ypred_all = zeros(nSamples,1);
    Ypred_vote = zeros(nSamples,1);
    found_fault_vote = false;
    first_detection_vote = NaN;

    for s = 1:nSamples
        window = zeros(windowLen, size(mat_signal,2));
        if s < windowLen
            window(end-s+1:end,:) = mat_signal(1:s,:);
        else
            window(:,:) = mat_signal(s-windowLen+1:s,:);
        end

        Xwin = build_features_multi(window, fs, windowLen, windowLen);
        Xwin_norm = (Xwin - mu) ./ sigma;
        Yprob = predict(net, Xwin_norm);
        [~, idxmax] = max(Yprob,[],2);
        Ypred_all(s) = idxmax-1;
    end

    % mocniejsze wygładzenie predykcji
    Ypred_all_filtered = medfilt1(Ypred_all,100);       % filtr medianowy
    Ypred_all_filtered = movmean(Ypred_all_filtered,80); % opcjonalne dodatkowe wygładzenie średnią
    Ypred_all_filtered = Ypred_all_filtered > 0.5;    % progowanie

    % ---- Głosowanie ----
    for s = 1:nSamples
        if s < tailStart
            Ypred_vote(s) = 0; % ignorujemy próbki przed tailStart
        else
            voteBuffer = [voteBuffer(2:end); Ypred_all_filtered(s)];
            if sum(voteBuffer) >= ceil(voteLen*0.7) % wyższy próg
                Ypred_vote(s) = 1;
            else
                Ypred_vote(s) = 0;
            end

            if ~found_fault_vote && Ypred_vote(s)==1
                first_detection_vote = (s-tailStart)*dt;
                found_fault_vote = true;
            end
        end
    end

    % ---- Wykresy w jednym figure z 2 subplotami ----
    t_plot = (0:nSamples-1)*dt;
    torque_plot = mat_signal(:,6);

    hFig = figure('Visible','off');

    subplot(2,1,1)
    yyaxis left
    plot(t_plot, torque_plot,'b','LineWidth',1.2); grid on;
    ylabel('Torque [Nm]');
    yyaxis right
    stairs(t_plot, Ypred_all,'r','LineWidth',1.5); ylim([-0.1 1.1]);
    ylabel('Detekcja bez głosowania');
    xlabel('Time [s]');
    title(sprintf('%s - predykcja bez głosowania', case_label));

    % ---- Dodanie linii pionowej pierwszego wykrycia bez głosowania ----
    idx_fault_novote = find(Ypred_all==1 & (1:nSamples)'>=tailStart,1,'first');
    if ~isempty(idx_fault_novote)
        xline(t_plot(idx_fault_novote),'--r','LineWidth',1.2,'Label','Wykrycie bez głosowania','LabelOrientation','horizontal');
    end

    subplot(2,1,2)
    yyaxis left
    plot(t_plot, torque_plot,'b','LineWidth',1.2); grid on;
    ylabel('Torque [Nm]');
    yyaxis right
    stairs(t_plot, Ypred_vote,'r','LineWidth',1.5); ylim([-0.1 1.1]);
    ylabel('Detekcja z głosowaniem');
    xlabel('Time [s]');
    if found_fault_vote
        title(sprintf('Pierwsze wykrycie (głosowanie): %.2f ms', first_detection_vote*1000));
    else
        title('Brak wykrycia błędu (głosowanie)');
    end

    saveas(hFig, fullfile(plot_dir,[case_label,'_plot.png']));
    close(hFig);

    fprintf('Zapisano wykres z głosowaniem i bez głosowania: %s\n', case_label);

    % ---- Macierz pomyłek dla każdego przypadku ----
    Ytrue_case = [zeros(tailStart-1,1); ones(nSamples-tailStart+1,1)];
    hFigMatrix = figure('Visible','off');
    confusionchart(Ytrue_case,Ypred_vote,'Title',sprintf('Macierz pomyłek - %s', case_label),...
        'ColumnSummary','column-normalized','RowSummary','row-normalized');
    saveas(hFigMatrix, fullfile(matrix_dir, ['matrix_fault_', case_label, '.png']));
    close(hFigMatrix);

    fprintf('Zapisano macierz pomyłek: matrix_fault_%s.png\n', case_label);
end

fprintf('Wszystkie wykresy i macierze pomyłek zapisane.\n');

