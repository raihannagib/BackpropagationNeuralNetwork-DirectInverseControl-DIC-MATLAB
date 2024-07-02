%% filename : generateDataset.m

%% PLANT
%                               1     
% y(k) = ------------------------------------------------
%        (1 + y(k-1)^2)) - (0.25 * u(k)) - (0.3 * u(k-1))

%% Generate Dataset

% Data Sinusoidal
y = zeros(12000, 1);  
u = zeros(12000, 1);

for i = 1:500
    u(i) = sind(i) * 0.1;
end

for i = 501:1000
    u(i) = sind(i) * 0.2;
end

for i = 1001:1500
    u(i) = sind(i) * 0.3;
end

for i = 1501:2000
    u(i) = sind(i) * 0.4;
end

for i = 2001:2500
    u(i) = sind(i) * 0.5;
end

for i = 2501:3000
    u(i) = sind(i) * 0.6;
end

for i = 3001:3500
    u(i) = sind(i) * 0.7;
end

for i = 3501:4000
    u(i) = sind(i) * 0.8;
end

for i = 4001:4500
    u(i) = sind(i) * 0.9;
end

for i = 4501:5000
    u(i) = sind(i) * 1;
end

for i = 5001:5500
    u(i) = sind(i) * 1.1;
end

for i = 5501:6000
    u(i) = sind(i) * 1.2;
end

for i = 6001:6500
    u(i) = sind(i) * 1.3;
end

for i = 6501:7000
    u(i) = sind(i) * 1.4;
end

for i = 7001:12000
    u(i) = sind(i);
end

[u_row, u_column] = size(u);
[y_row, y_column] = size(y);

for k =1:y_row
    if k == 1
        y(k) = (1 / (1 + 0)) - (0.25 * u(k)) - (0.3 * 0);
    else 
        y(k) = (1 / (1 + y(k-1)^2)) - (0.25 * u(k)) - (0.3 * u(k-1));
    end
end

% INPUT
uk = zeros(u_row, 1);
uk_min1 = zeros(u_row, 1);
uk_min2 = zeros(u_row, 1);
uk_min3 = zeros(u_row, 1);
yk = zeros(y_row, 1);
yk_min1 = zeros(y_row, 1);
yk_min2 = zeros(y_row, 1);


for i = 1:u_row
    uk(i) = u(i);
    yk(i) = y(i);
end

for i = 1:(u_row - 1)
    uk_min1(i + 1) = u(i);
    yk_min1(i + 1) = y(i);
end

for i = 1:(u_row - 2)
    uk_min2(i + 2) = u(i);
    yk_min2(i + 2) = y(i); 
end

for i = 1:(u_row - 3)
    uk_min3(i + 3) = u(i);
end

dataset = cat(2, uk, uk_min1, uk_min2, yk_min1, yk_min2, yk);
dataset_inverse = cat(2, uk_min1, uk_min2, uk_min3, yk, yk_min1, yk_min2, uk);
dataset_inverse_cl = cat(2, uk_min1, uk_min2, uk_min3, yk_min1, yk_min2, uk);

dataTable = array2table(dataset);
dataTable = renamevars(dataTable, ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"], ["u(k)", "u(k-1)", "u(k-2)", "y(k-1)", "y(k-2)", "y(k)/output"]);


dataTable_inverse = array2table(dataset_inverse);
dataTable_inverse = renamevars(dataTable_inverse, ["dataset_inverse1", "dataset_inverse2", "dataset_inverse3", "dataset_inverse4", "dataset_inverse5", "dataset_inverse6", "dataset_inverse7"], ["u(k-1)", "u(k-2)", "u(k-3)", "y(k)", "y(k-1)", "y(k-2)", "u(k)/output"]);

dataTable_inverse_cl = array2table(dataset_inverse_cl);
dataTable_inverse_cl = renamevars(dataTable_inverse_cl, ["dataset_inverse_cl1", "dataset_inverse_cl2", "dataset_inverse_cl3", "dataset_inverse_cl4", "dataset_inverse_cl5", "dataset_inverse_cl6"], ["u(k-1)", "u(k-2)", "u(k-3)", "y(k-1)", "y(k-2)", "u(k)/output"]);


writetable(dataTable_inverse, 'dataset_inverse.xlsx','Sheet',1);
writetable(dataTable, 'dataset.xlsx','Sheet',1);
writetable(dataTable, 'dataset_inverse_cl.xlsx','Sheet',1);
