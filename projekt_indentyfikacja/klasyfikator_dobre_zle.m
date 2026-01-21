clc; clear; close all; %% Ustawienia użytkownika
rng(2)
warning('off', 'all');

if isempty(gcp('nocreate'))
    parpool; % Starts the default parallel pool
end

path_data = './Faults_database/';
errors = 1; % zmień tu listę przypadków błędów (np. 1:5)
fs = 20000; % częstotliwość próbkowania (używana przy cechach)
windowLen = 100; % długość okna (próbki)
hop = windowLen/10; % hop. Możesz ustawić mniejszy
tailStart = 20001; % indeks początku części "fault" w plikach
nCases = length(errors);

%% Wczytaj pierwszy plik żeby ustalić nTail i sprawdzić zgodność
S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
if numel(S0.ia) < tailStart
    error('Plik ma mniej niż %d próbek.', tailStart);
end
nTail = numel(S0.ia) - (tailStart-1);

% Prealokacja macierzy tail (rows = nTail, cols = nCases)
ia_fault = zeros(nTail, nCases);
ib_fault = zeros(nTail, nCases);
ic_fault = zeros(nTail, nCases);
ialfa_fault = zeros(nTail, nCases);
ibeta_fault = zeros(nTail, nCases);
motor_torque_fault = zeros(nTail, nCases);

%% ===== Oś czasu dla danych fault =====
dt = 1/fs;
t_fault = (0:nTail-1) * dt;

%% ===== Folder na wykresy danych treningowych =====
train_plot_dir = './training_data/';
if ~exist(train_plot_dir, 'dir')
    mkdir(train_plot_dir);
end

%% ===== Wykresy przykładowych danych fault do treningu =====
for i = 1:nCases
    case_number = errors(i);

    hFig = figure('Visible','off');

    subplot(3,1,1)
    plot(t_fault, ia_fault(:,i),'r', ...
         t_fault, ib_fault(:,i),'g', ...
         t_fault, ic_fault(:,i),'b','LineWidth',1.1);
    grid on;
    ylabel('Prądy [A]');
    title(['Fault ', num2str(case_number), ' — prądy fazowe']);
    legend('i_a','i_b','i_c');

    subplot(3,1,2)
    plot(t_fault, ialfa_fault(:,i),'m', ...
         t_fault, ibeta_fault(:,i),'c','LineWidth',1.1);
    grid on;
    ylabel('\alpha\beta');
    title('Składowe Clarke');
    legend('\alpha','\beta');

    subplot(3,1,3)
    plot(t_fault, motor_torque_fault(:,i),'k','LineWidth',1.2);
    grid on;
    ylabel('Moment [Nm]');
    xlabel('Czas [s]');
    title('Moment elektromagnetyczny');

    fname = fullfile(train_plot_dir, ...
        sprintf('fault%d_plot.png', case_number));
    saveas(hFig, fname);
    close(hFig);
end

fprintf('Zapisano wykresy danych treningowych do folderu: %s\n', train_plot_dir);

% Wczytaj pliki i zapisz tail
for i = 1:nCases
    case_number = errors(i);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', case_number, '_faultType0.mat'));

    % Sprawdź długości
    if numel(S.ia) < tailStart + nTail - 1
        error('Plik fault%d ma za mało próbek.', case_number);
    end

    ia_fault(:,i) = S.ia(tailStart:end,1);
    ib_fault(:,i) = S.ib(tailStart:end,1);
    ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1);
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1);
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

% "OK" (poprawne) sygnały — przyjmujemy z ostatnio wczytanego pliku S (albo wczytaj osobny plik)
ia_poprawne = S.ia(1:tailStart-1,1);
ib_poprawne = S.ib(1:tailStart-1,1);
ic_poprawne = S.ic(1:tailStart-1,1);
ialfa_poprawne = S.ialfa(1:tailStart-1,1);
ibeta_poprawne = S.ibeta(1:tailStart-1,1);
motor_torque_poprawne = S.motor_torque(1:tailStart-1,1);

%% Przygotowanie macierzy wejściowej do ML
% Scal sygnały: kolumny = [ia ib ic ialfa ibeta motor_torque]
fault = [ia_fault(:), ib_fault(:), ic_fault(:), ialfa_fault(:), ibeta_fault(:), motor_torque_fault(:)];

% Uwaga: powyżej flattenujemy wszystkie kolumny jedna po drugiej — ale zamiast tego
% skorzystamy z funkcji build_features która pracuje na macierzy Nx3. Dostosujemy ją do 6-kanałowej.

% Zbuduj cechy z okien (adaptowana funkcja do 6 kanałów)
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
            
            % Combine stats and FFT for this specific channel
            f_all_channels = [f_all_channels, f_stats, domAmp];
        end
        
        % --- Inter-phase calculations (using first 3 columns: ia, ib, ic) ---
        ia = w(:,1); ib = w(:,2); ic = w(:,3);
        zero_seq = mean(ia + ib + ic);
        pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));
        
        % Combine all channel features with inter-phase features
        X = [X; f_all_channels, zero_seq, pos_seq];
    end
end

% Przygotuj macierze wejściowe dla fault i ok
% Najpierw zrekonstruujemy macierze N x C dla jednego przypadku: kolumny są poskładane w ia_fault(:,i) itd.
% Dla każdego case tworzymy macierz N x 6 i połączymy w długi zbiór (każdy case daje wiele okien)
X_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf];
end

% Z ok danych (przyjmujemy pojedynczy komplet poprawnych kanałów)
mat_ok = [ia_poprawne, ib_poprawne, ic_poprawne, ialfa_poprawne, ibeta_poprawne, motor_torque_poprawne];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

% Etykiety
Y_fault = ones(size(X_fault,1),1);
Y_ok = zeros(size(X_ok,1),1);

% Połącz i przetasuj
Xall = [X_fault; X_ok];
Yall = [Y_fault; Y_ok];
perm = randperm(size(Xall,1));
Xall = Xall(perm,:);
Yall = Yall(perm);

% Podział train/test
cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:);
Ytrain = Yall(training(cv));
Xtest = Xall(test(cv),:);
Ytest = Yall(test(cv));

%% Normalizacja (zachowujemy parametry do późniejszej normalizacji testów)
mu = mean(Xtrain,1);
sigma = std(Xtrain,[],1);
sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma;
Xtest_norm = (Xtest - mu) ./ sigma;

%% Sieć neuronowa (prosty fully connected deep net)
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

% Przygotuj dane do trenowania (tabela lub ds)
Ytrain_cat = categorical(Ytrain);
dsTrain = arrayDatastore(Xtrain_norm,'IterationDimension',1);
trainTbl = combine(dsTrain, arrayDatastore(Ytrain_cat));

% Trenuj
net = trainNetwork(Xtrain_norm, Ytrain_cat, layers, options);

%% Ocena na zbiorze testowym
Ypred_probs = predict(net, Xtest_norm); % Nx2 macierz prawdopodobieństw
%% ===== Folder na wykresy prawdopodobieństwa =====
prob_plot_dir = './training_data/probability/';
if ~exist(prob_plot_dir, 'dir')
    mkdir(prob_plot_dir);
end

[~, idxmax] = max(Ypred_probs, [], 2);
Ypred = idxmax - 1; % klasy 0/1

% Obliczanie macierzy pomyłek
confMat = confusionmat(Ytest, Ypred);
%% ===== Zapis macierzy pomyłek (PNG) + statystyki + AUC =====
matrix_dir = './matrix/';
if ~exist(matrix_dir, 'dir')
    mkdir(matrix_dir);
end

% numer błędu (np. errors = 1 albo 1:5)
case_number = errors(1);   % jeśli masz wiele – tu logicznie jest zbiorcza macierz

% ===== Wykres macierzy pomyłek (do zapisu) =====
hFig = figure('Visible','off');
confusionchart(Ytest, Ypred, ...
    'Title', sprintf('Macierz pomyłek dla błędu %d', case_number), ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

fname = fullfile(matrix_dir, ...
    sprintf('confusion_matrix_%d.png', case_number));

saveas(hFig, fname);
close(hFig);

fprintf('Macierz pomyłek zapisana do pliku: %s\n', fname);

% ===== (Opcjonalnie) Wyświetlenie na ekran =====
figure;
confusionchart(Ytest, Ypred, ...
    'Title', sprintf('Macierz pomyłek dla błędu %d', case_number), ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

% ===== Statystyki =====
acc = sum(Ypred == Ytest) / numel(Ytest);
fprintf('Accuracy = %.2f%%\n', acc * 100);

% ===== AUC =====
scoresPos = Ypred_probs(:, 2);   % prawdopodobieństwo klasy FAULT
[Xroc, Yroc, ~, AUC] = perfcurve(Ytest, scoresPos, 1);
fprintf('AUC = %.3f\n', AUC);


%% Bulk Processing with Multicore and GPU Support
plot_dir = './plots_binary/';
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

files = dir(fullfile(path_data, '*_faultType0.mat'));
windowLen_samples = windowLen; 
dt = 1/fs; 

% Capture variables for parfor
mu_val = mu;
sigma_val = sigma;
% If you have a compatible GPU, move the net to GPU memory
% Note: trainNetwork already returns a DAGNetwork/SeriesNetwork that 
% can use the GPU automatically if available during predict().
net_gpu = net; 

fprintf('================================== \n')
fprintf('Starting parallel bulk plot generation...\n')

all_files_latencies = nan(length(files), 1);

parfor f = 1:length(files)
    filename = files(f).name;
    case_label = erase(filename, '_faultType0.mat');

    % Use a local struct to avoid transparency issues in parfor
    S_curr = load(fullfile(path_data, filename));
    mat_signal = [S_curr.ia, S_curr.ib, S_curr.ic, S_curr.ialfa, S_curr.ibeta, S_curr.motor_torque];

    nSamples = size(mat_signal,1);
    nChannels = size(mat_signal,2);
    Ypred_all = zeros(nSamples,1);

    found_fault_local = false;
    det_time_local = NaN;

    % --- Online Simulation (Processing) ---
    for s = 1:nSamples
        window = zeros(windowLen_samples, nChannels);
        if s < windowLen_samples
            window(end-s+1:end, :) = mat_signal(1:s, :);
        else
            window(:, :) = mat_signal(s-windowLen_samples+1:s, :);
        end

        % Feature Extraction
        Xwin = build_features_multi(window, fs, windowLen_samples, windowLen_samples);
        Xwin_norm = (Xwin - mu_val) ./ sigma_val;

        % Prediction (Inference)
        % Using 'ExecutionEnvironment', 'gpu' if available
        Yprob = predict(net_gpu, Xwin_norm, 'ExecutionEnvironment', 'auto');
        [~, idxmax] = max(Yprob,[],2);
        Ypred_all(s) = idxmax - 1; 
        
        if ~found_fault_local && Ypred_all(s) == 1 && s >= tailStart
            det_time_local = (s - tailStart) * dt;
            found_fault_local = true;
        end

        all_files_latencies(f) = det_time_local;
    end

    % --- Generate and Save Plot ---
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

%% Pobierz sygnał
mat_signal = [S.ia, S.ib, S.ic, S.ialfa, S.ibeta, S.motor_torque];
nSamples = size(mat_signal,1);
nChannels = size(mat_signal,2);

%% Tablica wyników klasyfikacji (dla każdej próbki)
Ypred_all = zeros(nSamples,1);   % 0 – brak błędu, 1 – błąd

found_fault = false;
first_detection_time = NaN;
fault_start_idx = tailStart;

%% Parametry głosowania
voteLen = 10;                     % długość okna głosowania
voteBuffer = zeros(voteLen,1);    % bufor głosów
Ypred_vote = zeros(nSamples,1);   % wyniki po głosowaniu

found_fault_vote = false;
first_detection_time_vote = NaN;

%% Główna pętla z klasyfikacją
for s = 1:nSamples

    % ===== Budowa okna =====
    window = zeros(windowLen_samples, nChannels);
    if s < windowLen_samples
        window(end-s+1:end, :) = mat_signal(1:s, :);
    else
        window(:, :) = mat_signal(s-windowLen_samples+1:s, :);
    end

    % ===== Ekstrakcja cech =====
    Xwin = build_features_multi(window, fs, windowLen_samples, windowLen_samples);

    % ===== Normalizacja =====
    Xwin_norm = (Xwin - mu) ./ sigma;

    % ===== Predykcja =====
    Yprob = predict(net, Xwin_norm);
    [~, idxmax] = max(Yprob,[],2);
    Ypred = idxmax - 1;

    % ===== Zapis wyniku =====
    Ypred_all(s) = Ypred;

    % ================= GŁOSOWANIE =================
    % przesuwamy bufor i dodajemy nowy wynik
    voteBuffer = [voteBuffer(2:end); Ypred];

    % policz liczbę "1" w buforze
    nVotes = sum(voteBuffer);

    % jeśli liczba 1 >= 6 → głosujemy "1"
    if nVotes >= ceil(voteLen/2)
        Ypred_vote(s) = 1;
    else
        Ypred_vote(s) = 0;
    end

    % ===== Detekcja pierwszego błędu (głosowanie) =====
    if ~found_fault_vote && Ypred_vote(s) == 1 && s >= fault_start_idx
        first_detection_time_vote = (s - fault_start_idx) * dt;
        found_fault_vote = true;
    end

    % ===== Detekcja pierwszego błędu (oryginalna) =====
    if ~found_fault && Ypred == 1 && s >= fault_start_idx
        first_detection_time = (s - fault_start_idx) * dt;
        found_fault = true;
    end

end

if ~found_fault
    fprintf('Błąd nie został wykryty przy metodzie ORYGINALNEJ.\n');
end
if ~found_fault_vote
    fprintf('Błąd nie został wykryty przy metodzie GŁOSOWANIA.\n');
end

%% Czas osi
t = (0:nSamples-1) * dt;
torque = mat_signal(:,6);

%% ===== WYKRESY =====
figure;

subplot(2,1,1);
yyaxis left
plot(t, torque, 'b', 'LineWidth', 1.2);
ylabel('Torque [Nm]'); grid on;

yyaxis right
stairs(t, Ypred_all, 'r', 'LineWidth', 1.5);
ylabel('Detekcja (0 / 1)');
ylim([-0.1 1.1]);
title('Oryginalna detekcja błędu');
xlabel('Czas [s]');
legend('Torque','Detekcja ORYGINALNA','Location','best');

subplot(2,1,2);
yyaxis left
plot(t, torque, 'b', 'LineWidth', 1.2);
ylabel('Torque [Nm]'); grid on;

yyaxis right
stairs(t, Ypred_vote, 'm', 'LineWidth', 1.5);
ylabel('Detekcja (0 / 1)');
ylim([-0.1 1.1]);
title('Detekcja błędu z GŁOSOWANIEM');
xlabel('Czas [s]');
legend('Torque','Detekcja z GŁOSOWANIEM','Location','best');

%% ===== Raport czasów detekcji =====
fprintf('\n=================== CZASY DETEKCJI ===================\n');

if found_fault
    fprintf('Pierwsze wykrycie (oryginalne):      %.2f ms\n', first_detection_time * 1000);
else
    fprintf('Brak wykrycia metodą oryginalną.\n');
end

if found_fault_vote
    fprintf('Pierwsze wykrycie (głosowanie):      %.2f ms\n', first_detection_time_vote * 1000);
else
    fprintf('Brak wykrycia metodą głosowania.\n');
end

fprintf('=====================================================\n');
