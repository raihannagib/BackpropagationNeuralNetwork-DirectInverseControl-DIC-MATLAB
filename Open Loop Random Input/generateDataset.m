%% filename : generateDataset.m

%% PLANT
%              1     
% y(k) = ---------------- + (0.25 * u(k)) - (0.3 * u(k-1))
%        (1 + y(k-1)^2)) 

%% Generate Dataset

% Data Random
y = zeros(50000, 1);                
u = (1 + 1) * rand(50000, 1) - 1;

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

dataTable = array2table(dataset);
dataTable = renamevars(dataTable, ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"], ["u(k)", "u(k-1)", "u(k-2)", "y(k-1)", "y(k-2)", "y(k)/output"]);


dataTable_inverse = array2table(dataset_inverse);
dataTable_inverse = renamevars(dataTable_inverse, ["dataset_inverse1", "dataset_inverse2", "dataset_inverse3", "dataset_inverse4", "dataset_inverse5", "dataset_inverse6", "dataset_inverse7"], ["u(k-1)", "u(k-2)", "u(k-3)", "y(k)", "y(k-1)", "y(k-2)", "u(k)/output"]);

writetable(dataTable_inverse, 'dataset_inverse.xlsx','Sheet',1);
writetable(dataTable, 'dataset.xlsx','Sheet',1);

