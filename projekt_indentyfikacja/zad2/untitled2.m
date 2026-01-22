clc; clear; close all;

%% ===== Ustawienia użytkownika =====
rng(2)
warning('off','all');

if isempty(gcp('nocreate'))
    parpool; % starts parallel pool
end

path_data = './Faults_database/';
errors = 1:63;      % wszystkie przypadki błędów
fs = 20000;         % częstotliwość próbkowania
windowLen = 1000;    % długość okna (próbki)
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

%% ===== Folder na training plots =====
train_plot_dir = './training_data_2/';
if ~exist(train_plot_dir,'dir')
    mkdir(train_plot_dir);
end

%% ===== Funkcja rysująca przykładowe okno =====
function plot_example_window(mat, Xf, windowLen, hop, fs, title_str, save_path)

    example_window_idx = 1; % pierwsze okno
    sample_window = mat((example_window_idx-1)*hop+1:(example_window_idx-1)*hop+windowLen, :);
    t_win = (0:windowLen-1)/fs;

    var_names = {'ia','ib','ic','ialfa','ibeta','torque'};
    nVars = numel(var_names);

    % ===== cechy dla przykładowego okna =====
    Xwin = Xf(example_window_idx, :);
    nFeatTotal = numel(Xwin);

    % ===== podział cech =====
    nFeatPerVar = 7;              % 6 * 7 = 42
    nMainFeats  = nVars*nFeatPerVar;

    if nFeatTotal < nMainFeats + 2
        error('Za mało cech: %d. Oczekiwano co najmniej %d.', ...
              nFeatTotal, nMainFeats + 2);
    end

    % ostatnie dwie cechy
    zero_seq = Xwin(end-1);
    pos_seq  = Xwin(end);

    % ===== Figure niewidoczne =====
    fig = figure( ...
        'Name', title_str, ...
        'NumberTitle','off', ...
        'Position',[100 100 1100 900], ...
        'Visible','off' );

    % ===== 1) Sygnały w czasie =====
    subplot(4,2,[1 2])
    plot(t_win, sample_window, 'LineWidth',1.2)
    xlabel('Czas [s]')
    ylabel('Amplituda')
    legend({'ia','ib','ic','ialfa','ibeta','motor\_torque'}, 'Location', 'best');
    title(['Przykładowe okno sygnału - ' title_str])
    grid on

    % ===== 2) Wykresy cech – osobno dla każdej zmiennej =====
    for v = 1:nVars
        idx_start = (v-1)*nFeatPerVar + 1;
        idx_end   = v*nFeatPerVar;

        feats_v = Xwin(idx_start:idx_end);

        subplot(4,2,2+v)
        bar(feats_v)
        xlabel('Indeks cechy')
        ylabel('Wartość')
        title(['Cechy - ' var_names{v}])
        grid on
    end

    % ===== 3) Tekst: zero_seq i pos_seq =====
    % umieszczony pod ostatnim subplotem
    annotation(fig, 'textbox', ...
        [0.1 0.02 0.8 0.05], ...   % [x y w h] w jednostkach znormalizowanych
        'String', sprintf('zero\\_seq = %.4g    |    pos\\_seq = %.4g', zero_seq, pos_seq), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'EdgeColor','none', ...
        'FontSize',11, ...
        'FontWeight','bold');

    % ===== Zapis i zamknięcie =====
    saveas(fig, save_path);
    close(fig)

end


%% ===== Przygotowanie i rysowanie danych =====
for i = 1:nCases
    case_number = errors(i);
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);

    save_path = fullfile(train_plot_dir, sprintf('case%d_example_window.png', case_number));
    plot_example_window(mat, Xf, windowLen, hop, fs, sprintf('Case %d', case_number), save_path);
end

%% ===== Przykład okna bez błędu =====
mat_ok = [ia_ok, ib_ok, ic_ok, ialfa_ok, ibeta_ok, motor_torque_ok];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

save_path = fullfile(train_plot_dir, 'ok_example_window.png');
plot_example_window(mat_ok, X_ok, windowLen, hop, fs, 'OK signal', save_path);
