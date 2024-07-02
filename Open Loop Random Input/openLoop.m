% PERCOBAAN OPEN LOOP
%                  --------------           --------------
%                 |              |         |              |
%                 |   Inverse    |   u(k)  |              |
%    y(k) ----- > |    Plant     | ----- > |    Plant     | ----- > y'(k)
%           |---> |              |   | |-> |              |---------|
%           |     |              |   | |   |              |         |
%           |      --------------    | |    --------------          |
%           |                        | |                            |
%           |------------------------| |----------------------------|                         |
%

clc;
clear;

% Impor data
dataTable_inverse = readtable('dataset_inverse.xlsx', 'sheet', 'Sheet1');

% DATA PREPROCESSING
% Penentuan feature dan target dari dataTable
feature_inverse = dataTable_inverse(:, 1:end-1);
target_inverse = dataTable_inverse(:, end);

% Normalisasi data feature
feature_inverse = normalize(feature_inverse, 'range', [-0.5, 0.5]);
target_inverse = normalize(target_inverse, 'range', [-0.5, 0.5]);

% Inisialisasi Variabel untuk Menampung nilai Y_referensi & Y_model

% Mencari tahu jumlah baris dan kolom dari feature
[feature_num_row_inverse, feature_num_column_inverse] = size(feature_inverse);

% Mencari tahu jumlah baris dan kolom dari target
[target_num_row_inverse, target_num_column_inverse] = size(target_inverse);


% Pemisahan Data Training, Data Validasi, dan Data Testing
X_train_inverse = table2array(feature_inverse(1:45000, :));
X_train = table2array(feature_inverse(1:45000, :));
y_train_inverse = table2array(target_inverse(1:45000, :));
y_train = table2array(feature_inverse(1:45000, 4));


X_test_inverse = table2array(feature_inverse(45000 + 1:end, :));
X_test = table2array(feature_inverse(45000 + 1:end, :));
y_test_inverse = table2array(target_inverse(45000 + 1:end, :));
y_test = table2array(feature_inverse(45000 + 1:end, 4));


% Inisisalisasi variabel penampung data aktual dan estimasi dari model
vektor_u_ANN_train = zeros(length(X_train_inverse), 1);
vektor_u_ref_train = table2array(target_inverse(1:45000, :));
vektor_y_ANN_train = zeros(length(X_train_inverse), 1);
vektor_y_ref_train = table2array(feature_inverse(1:45000, 4));

vektor_u_ANN_test = zeros(length(X_test_inverse), 1);
vektor_u_ref_test = table2array(target_inverse(45000 + 1:end, :));
vektor_y_ANN_test = zeros(length(X_test_inverse), 1);
vektor_y_ref_test = table2array(feature_inverse(45000 + 1:end, 4));


% Model BPNN
% Inisialisasi jumlah input layer, hidden layer, dan output layer model ANN
% Inverse
n_inverse = feature_num_column_inverse;             % input layer
p_inverse = 10;                                     % hidden layer 
m_inverse = target_num_column_inverse;              % output layer

% Inisialisasi jumlah input layer, hidden layer, dan output layer model ANN
% Plant
n = feature_num_column_inverse - 1;                             % input layer
p = 10;                                                     % hidden layer 
m = target_num_column_inverse;                              % output layer


% Impor bobot dan bias yang sudah dilatih dari training model ANN Plant dan
% model ANN Inverse
V = readmatrix("bobot_V.xlsx");
W = readmatrix("bobot_W.xlsx");
V_0j = readmatrix("bias_V.xlsx");
W_0k = readmatrix("bias_W.xlsx");

V_inverse = readmatrix("bobot_V_inverse.xlsx");
W_inverse = readmatrix("bobot_W_inverse.xlsx");
V_0j_inverse = readmatrix("bias_V_inverse.xlsx");
W_0k_inverse = readmatrix("bias_W_inverse.xlsx");

% Inisialisasi nilai lama
W_jk_lama = 0;
W_0k_lama = 0;
V_ij_lama = 0;
V_0j_lama = 0;

% Algoritma BPNN
% Penentuan nilai parameter iterasi
iterasi = 5000;                 % JUMLAH EPOCH
iter = 0;
iter_val = 0;
Ep_stop = 1;
alpha = 0.5;                    % LAJU PEMBELAJARAN 
miu = 0.5;                      % LAJU PEMBELAJARAN MOMENTUM 



% Inisialisasi nilai lama
W_jk_lama_inverse = 0;
W_0k_lama_inverse = 0;
V_ij_lama_inverse = 0;
V_0j_lama_inverse = 0;


% Training data
while Ep_stop == 1 && iter < iterasi
    iter = iter + 1;
    for a = 3:length(X_train_inverse)
        % ==================== PROSES FEEDFORWARD ====================
        % ========================= INVERSE ==========================
        % Menghitung semua sinyal input dengan bobotnya
        z_inj_inverse_train = V_0j_inverse + X_train_inverse(a,:) * V_inverse;

        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for j = 1:p_inverse
            zj_inverse_train(1,j) = -1 + 2 / (1 + exp(-z_inj_inverse_train(1, j)));
        end

        % Menghitung semua sinyal input dengan bobotnya
        y_ink_inverse_train = W_0k_inverse + zj_inverse_train * W_inverse;
        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for r = 1:m_inverse
            y_k_inverse_train(a,r) = -1 + 2 / (1 + exp(-y_ink_inverse_train(1, r)));
        end

        vektor_u_ANN_train(a, 1) = y_k_inverse_train(a, 1); 

        % Menghitung nilai error
        E_inverse_train(1, a) = abs(y_train_inverse(a,:) - y_k_inverse_train(a));

        % Menghitung nilai total error kuadratik (MSE) data validasi
        E_mse_inverse_train(1, a) = (y_train_inverse(a, :) - y_k_inverse_train(a)).^2;

        % ==================== PROSES FEEDFORWARD ====================
        % ========================== PLANT ===========================
        if a == 3
            feature_train = [vektor_u_ANN_train(a, 1) X_train(a, 1) X_train(a, 2) X_train(a, 5) X_train(a, 6)];
        elseif a == 4
            feature_train = [vektor_u_ANN_train(a, 1) vektor_u_ANN_train(a - 1, 1) X_test(a, 2) X_train(a, 5) X_train(a, 6)];
        else
            feature_train = [vektor_u_ANN_train(a, 1) vektor_u_ANN_train(a - 1, 1) vektor_u_ANN_train(a - 2, 1) X_train(a, 5) X_train(a, 6)]; 
        end

        % Menghitung semua sinyal input dengan bobotnya
        z_inj_train = V_0j + feature_train * V;

        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for j = 1:p
            zj_train(1,j) = -1 + 2 / (1 + exp(-z_inj_train(1, j)));
        end

        % Menghitung semua sinyal input dengan bobotnya
        y_ink_train = W_0k + zj_train * W;
        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for r = 1:m
            y_k_train(a,r) = -1 + 2 / (1 + exp(-y_ink_train(1, r)));
        end

        vektor_y_ANN_train(a, 1) = y_k_train(a, 1);

        % Menghitung nilai error
        E_train(1, a) = abs(y_train(a,:) - y_k_train(a));

        % Menghitung nilai total error kuadratik (MSE) data validasi
        E_mse_train(1, a) = (y_train(a, :) - y_k_train(a)).^2;


        if a < length(X_train_inverse)
            X_train_inverse(a + 1, 1) = vektor_u_ANN_train(a, 1);
        end

        if a < length(X_train_inverse)
            X_train(a + 1, 5) = vektor_y_ANN_train(a, 1);
        end

        if a < length(X_train_inverse)
            X_train_inverse(a + 1, 2) = vektor_u_ANN_train(a - 1, 1);
        end

        if a < length(X_train_inverse)
            X_train_inverse(a + 1, 3) = vektor_u_ANN_train(a - 2, 1);
        end

        if a < length(X_train_inverse)
            X_train(a + 1, 6) = vektor_y_ANN_train(a - 1, 1);
        end
        
        
        % ==================== PROSES BACKPROPAGATION ====================
        % =========================== INVERSE ============================
        % Menghitung informasi error
        do_k_inverse = (y_train_inverse(a,:) - y_k_inverse_train(a)) .* ((1 + y_k_inverse_train(a)) * (1 - y_k_inverse_train(a)) / 2);
        % Menghitung besarnya koreksi bobot unit output
        w_jk_inverse = alpha * zj_inverse_train' * do_k_inverse + miu * W_jk_lama_inverse;
        % Menghitung besarnya koreksi bias output
        w_0k_inverse = alpha * do_k_inverse + miu * W_0k_lama_inverse;
        
        W_jk_lama_inverse = w_jk_inverse;
        W_0k_lama_inverse = w_0k_inverse;
        
        % Menghitung semua koreksi error
        do_inj_inverse  = do_k_inverse * W_inverse';
        % Menghitung nilai aktivasi koreksi error
        do_j_inverse    = do_inj_inverse .* ((1 + zj_inverse_train) .* (1 - zj_inverse_train) / 2);
        % Menghtiung koreksi bobot unit hidden
        v_ij_inverse    = alpha * X_train_inverse(a,:)' * do_j_inverse + miu * V_ij_lama_inverse;
        % Menghitung koreksi error bias unit hidden
        v_0j_inverse    = alpha * do_j_inverse + miu * V_0j_lama_inverse;
        
        V_ij_lama_inverse = v_ij_inverse;
        V_0j_lama_inverse = v_0j_inverse;
        
        % Menng-update bobot dan bias untuk setiap unit output
        W_inverse = W_inverse + w_jk_inverse;
        W_0k_inverse = W_0k_inverse + w_0k_inverse;
        
        % Mengupdate bobot dan bias untuk setiap unit hidden
        V_inverse = V_inverse + v_ij_inverse;
        V_0j_inverse = V_0j_inverse + v_0j_inverse;
        
        % ==================== PROSES BACKPROPAGATION ====================
        % ============================ PLANT =============================
        % Menghitung informasi error
        do_k = (y_train(a,:) - y_k_train(a)) .* ((1 + y_k_train(a)) * (1 - y_k_train(a)) / 2);
        % Menghitung besarnya koreksi bobot unit output
        w_jk = alpha * zj_train' * do_k + miu * W_jk_lama;
        % Menghitung besarnya koreksi bias output
        w_0k = alpha * do_k + miu * W_0k_lama;
        
        W_jk_lama = w_jk;
        W_0k_lama = w_0k;
        
        % Menghitung semua koreksi error
        do_inj  = do_k * W';
        % Menghitung nilai aktivasi koreksi error
        do_j    = do_inj .* ((1 + zj_train) .* (1 - zj_train) / 2);
        % Menghtiung koreksi bobot unit hidden
        v_ij    = alpha * feature_train' * do_j + miu * V_ij_lama;
        % Menghitung koreksi error bias unit hidden
        v_0j    = alpha * do_j + miu * V_0j_lama;
        
        V_ij_lama = v_ij;
        V_0j_lama = v_0j;
        
        % Menng-update bobot dan bias untuk setiap unit output
        W = W + w_jk;
        W_0k = W_0k + w_0k;
        
        % Mengupdate bobot dan bias untuk setiap unit hidden
        V = V + v_ij;
        V_0j = V_0j + v_0j;
    end
    
    % Menghitung nilai error training pada tiap epoch
    Ep_inverse(1, iter) = sum(E_train) / length(X_train_inverse);
    Ep(1, iter) = sum(E_inverse_train) / length(X_train_inverse);
    
    % Menghitung nilai total error kuadratik (MSE) pada tiap epoch
    MSE_inverse_train(iter, 1) = sum(E_mse_inverse_train) / length(X_train_inverse);
    MSE_train(iter, 1) = sum(E_mse_train) / length(X_train_inverse);
end

MSE_inverse_train_total = sum(MSE_inverse_train) / length(MSE_inverse_train);
MSE_train_total = sum(MSE_train) / length(MSE_train);


% Testing Data
for a = 3:length(X_test_inverse)
    % ==================== PROSES FEEDFORWARD ====================
    % ========================= INVERSE ==========================
    % Menghitung semua sinyal input dengan bobotnya
    z_inj_inverse_test = V_0j_inverse + X_test_inverse(a,:) * V_inverse;

    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for j = 1:p_inverse
        zj_inverse_test(1,j) = -1 + 2 / (1 + exp(-z_inj_inverse_test(1, j)));
    end

    % Menghitung semua sinyal input dengan bobotnya
    y_ink_inverse_test = W_0k_inverse + zj_inverse_test * W_inverse;
    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for r = 1:m_inverse
        y_k_inverse_test(a,r) = -1 + 2 / (1 + exp(-y_ink_inverse_test(1, r)));
    end
    
    vektor_u_ANN_test(a, 1) = y_k_inverse_test(a, 1); 

    % Menghitung nilai error
    E_inverse_test(1, a) = abs(y_test_inverse(a,:) - y_k_inverse_test(a));

    % Menghitung nilai total error kuadratik (MSE) data validasi
    E_mse_inverse_test(1, a) = (y_test_inverse(a, :) - y_k_inverse_test(a)).^2;
    
    % ==================== PROSES FEEDFORWARD ====================
    % ========================== PLANT ===========================
    if a == 3
        feature_test = [vektor_u_ANN_test(a, 1) X_test(a, 1) X_test(a, 2) X_test(a, 5) X_test(a, 6)];
    elseif a == 4 
        feature_test = [vektor_u_ANN_test(a, 1) vektor_u_ANN_test(a - 1, 1) X_test(a, 2) X_test(a, 5) X_test(a, 6)];
    else
        feature_test = [vektor_u_ANN_test(a, 1) vektor_u_ANN_test(a - 1, 1) vektor_u_ANN_test(a - 2, 1) X_test(a, 5) X_test(a, 6)]; 
    end
    
    % Menghitung semua sinyal input dengan bobotnya
    z_inj_test = V_0j + feature_test * V;

    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for j = 1:p
        zj_test(1,j) = -1 + 2 / (1 + exp(-z_inj_test(1, j)));
    end

    % Menghitung semua sinyal input dengan bobotnya
    y_ink_test = W_0k + zj_test * W;
    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for r = 1:m
        y_k_test(a,r) = -1 + 2 / (1 + exp(-y_ink_test(1, r)));
    end
    
    vektor_y_ANN_test(a, 1) = y_k_test(a, 1);

    % Menghitung nilai error
    E_test(1, a) = abs(y_test(a,:) - y_k_test(a));

    % Menghitung nilai total error kuadratik (MSE) data validasi
    E_mse_test(1, a) = (y_test(a, :) - y_k_test(a)).^2;
    

   
    if a < length(X_test_inverse)
        X_test_inverse(a + 1, 1) = vektor_u_ANN_test(a, 1);
    end

    if a < length(X_test_inverse)
        X_test_inverse(a + 1, 2) = vektor_u_ANN_test(a - 1, 1);
    end
    
    if a < length(X_train_inverse)
        X_inverse_inverse(a + 1, 3) = vektor_u_ANN_test(a - 2, 1);
    end
    
    
    if a < length(X_train_inverse)
        X_test(a + 1, 5) = vektor_y_ANN_test(a, 1);
    end

    if a < length(X_train_inverse)
        X_test(a + 1, 6) = vektor_y_ANN_test(a - 1, 1);
    end
end

Ep_inverse_test = sum(E_inverse_test) / length(X_test_inverse);
MSE_inverse_test = sum(E_mse_inverse_test) / length(X_test_inverse);
Ep_test = sum(E_test) / length(X_test_inverse);
MSE_test = sum(E_mse_test) / length(X_test_inverse);

figure;
plot(vektor_y_ref_test, 'r-', 'Linewidth', 1);
hold on
plot(vektor_y_ANN_test, 'b-', 'LineWidth', 1);
hold off
ylabel('Value'); xlabel('Sample Data');
legend("y(k) referensi testing", "y(k) keluaran testing");

figure;
plot(vektor_u_ref_test, 'r-', 'Linewidth', 1);
hold on
plot(vektor_u_ANN_test, 'b-', 'LineWidth', 1);
hold off
ylabel('Value'); xlabel('Sample Data');
legend("u(k) referensi testing", "u(k) keluaran testing");

figure;
plot(vektor_y_ref_train, 'r-', 'Linewidth', 1);
hold on
plot(vektor_y_ANN_train, 'b-', 'LineWidth', 1);
hold off
ylabel('Value'); xlabel('Sample Data');
legend("y(k) referensi traning", "y(k) keluaran training");

figure;
plot(vektor_u_ref_train, 'r-', 'Linewidth', 1);
hold on
plot(vektor_u_ANN_train, 'b-', 'LineWidth', 1);
hold off
ylabel('Value'); xlabel('Sample Data');
legend("u(k) referensi training", "u(k) keluaran training");


x = 1:5000;
y = 1:45000;

figure;
scatter(x, vektor_y_ref_test, 'Linewidth', 1);
hold on
scatter(x, vektor_y_ANN_test, 'Linewidth', 1);
legend("y(k) referensi testing", "y(k) keluaran testing");
hold off


figure;
scatter(x, vektor_u_ref_test, 'Linewidth', 1);
hold on
scatter(x, vektor_u_ANN_test, 'Linewidth', 1);
legend("u(k) referensi testing", "u(k) keluaran testing");
hold off


figure;
scatter(y, vektor_y_ref_train, 'Linewidth', 1);
hold on
scatter(y, vektor_y_ANN_train, 'Linewidth', 1);
legend("y(k) referensi training", "y(k) keluaran training");
hold off


figure;
scatter(y, vektor_u_ref_train, 'Linewidth', 1);
hold on
scatter(y, vektor_u_ANN_train, 'Linewidth', 1);
legend("u(k) referensi training", "u(k) keluaran training");
hold off

disp("MSE Inverse train = " + MSE_inverse_train_total);
disp("MSE train = " + MSE_train_total);
disp("MSE Inverse test = " + MSE_inverse_test);
disp("MSE test = " + MSE_test);
