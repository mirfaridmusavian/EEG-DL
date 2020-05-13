clear all
clc

format long

%%
% Read the Data and Create Dataset
Stack_Dataset = readmatrix("Dataset.csv");
Labels = readmatrix("Labels.csv");

%% Compute Covariance Matrix
covariance_matrix = cov(Stack_Dataset);
xlswrite('covariance_matrix.xlsx', covariance_matrix);

figure(1)
imagesc(covariance_matrix)
axis square
title('Covariance Matrix for Subject One', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Covariance_Matrix_for_Subject_One', '-dpng',  '-r600')

%% Compute Pearson Matrix
Pearson_matrix = corrcoef(Stack_Dataset);
xlswrite('Pearson_matrix.xlsx', Pearson_matrix);

figure(2)
imagesc(Pearson_matrix)
axis square
title('Pearson Matrix for Subject One', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Pearson_matrix_for_Subject_One', '-dpng',  '-r600')

%% Compute Absolute Pearson Matrix
Absolute_Pearson_matrix = abs(Pearson_matrix);
xlswrite('Absolute_Pearson_matrix.xlsx', Absolute_Pearson_matrix);

figure(3)
imagesc(Absolute_Pearson_matrix)
axis square
title('Absolute Pearson Matrix for Subject Three', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Absolute_Pearson_matrix_for_Subject_One', '-dpng',  '-r600')

%% Compute Adjacency Matrix
Eye_Matrix = eye(44, 44);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
xlswrite('Adjacency_Matrix.xlsx', Adjacency_Matrix);

figure(4)
imagesc(Adjacency_Matrix)
axis square
title('Adjacency Matrix for Subject One', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Adjacency_Matrix_for_Subject_One', '-dpng',  '-r600')

%%
All_Data = [Stack_Dataset, Labels];
rowrank = randperm(size(All_Data, 1));
All_Dataset = All_Data(rowrank, :);
[row, ~] = size(All_Dataset);

training_set   = All_Dataset(1:fix(row/10*9),     1:44);
test_set       = All_Dataset(fix(row/10*9)+1:end, 1:44);

training_label = All_Dataset(1:fix(row/10*9),     end);
test_label     = All_Dataset(fix(row/10*9)+1:end, end);

xlswrite('training_set.xlsx', training_set);
xlswrite('test_set.xlsx', test_set);
xlswrite('training_label.xlsx', training_label);
xlswrite('test_label.xlsx', test_label);
