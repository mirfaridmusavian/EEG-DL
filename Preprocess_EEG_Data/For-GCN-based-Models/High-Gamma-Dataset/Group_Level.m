clear all
clc

format long

%%
Dataset_1 = readmatrix("Subject_Dataset/Subject_1/Dataset.csv");
Dataset_2 = readmatrix("Subject_Dataset/Subject_2/Dataset.csv");
Dataset_3 = readmatrix("Subject_Dataset/Subject_3/Dataset.csv");
Dataset_4 = readmatrix("Subject_Dataset/Subject_4/Dataset.csv");
Dataset_5 = readmatrix("Subject_Dataset/Subject_5/Dataset.csv");
Dataset_6 = readmatrix("Subject_Dataset/Subject_6/Dataset.csv");
Dataset_7 = readmatrix("Subject_Dataset/Subject_7/Dataset.csv");
Dataset_8 = readmatrix("Subject_Dataset/Subject_8/Dataset.csv");
Dataset_9 = readmatrix("Subject_Dataset/Subject_9/Dataset.csv");
Dataset_10 = readmatrix("Subject_Dataset/Subject_10/Dataset.csv");
Dataset_11 = readmatrix("Subject_Dataset/Subject_11/Dataset.csv");
Dataset_12 = readmatrix("Subject_Dataset/Subject_12/Dataset.csv");
Dataset_13 = readmatrix("Subject_Dataset/Subject_13/Dataset.csv");
Dataset_14 = readmatrix("Subject_Dataset/Subject_14/Dataset.csv");

Stack_Dataset = [Dataset_1; Dataset_2; Dataset_3; Dataset_4; 
                 Dataset_5; Dataset_6; Dataset_7; Dataset_8;
                 Dataset_9; Dataset_10; Dataset_11; Dataset_12; 
                 Dataset_13; Dataset_14];
             
%%
Labels_1 = readmatrix("Subject_Dataset/Subject_1/Labels.csv");
Labels_2 = readmatrix("Subject_Dataset/Subject_2/Labels.csv");
Labels_3 = readmatrix("Subject_Dataset/Subject_3/Labels.csv");
Labels_4 = readmatrix("Subject_Dataset/Subject_4/Labels.csv");
Labels_5 = readmatrix("Subject_Dataset/Subject_5/Labels.csv");
Labels_6 = readmatrix("Subject_Dataset/Subject_6/Labels.csv");
Labels_7 = readmatrix("Subject_Dataset/Subject_7/Labels.csv");
Labels_8 = readmatrix("Subject_Dataset/Subject_8/Labels.csv");
Labels_9 = readmatrix("Subject_Dataset/Subject_9/Labels.csv");
Labels_10 = readmatrix("Subject_Dataset/Subject_10/Labels.csv");
Labels_11 = readmatrix("Subject_Dataset/Subject_11/Labels.csv");
Labels_12 = readmatrix("Subject_Dataset/Subject_12/Labels.csv");
Labels_13 = readmatrix("Subject_Dataset/Subject_13/Labels.csv");
Labels_14 = readmatrix("Subject_Dataset/Subject_14/Labels.csv");

Labels = [Labels_1; Labels_2; Labels_3; Labels_4; 
          Labels_5; Labels_6; Labels_7; Labels_8;
          Labels_9; Labels_10; Labels_11; Labels_12; 
          Labels_13; Labels_14];
             
%% Compute Covariance Matrix
covariance_matrix = cov(Stack_Dataset);
xlswrite('covariance_matrix.xlsx', covariance_matrix);

figure(1)
imagesc(covariance_matrix)
axis square
title('Covariance Matrix for 14 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Covariance_Matrix_for_14_Subjects', '-dpng',  '-r600')

%% Compute Pearson Matrix
Pearson_matrix = corrcoef(Stack_Dataset);
xlswrite('Pearson_matrix.xlsx', Pearson_matrix);

figure(2)
imagesc(Pearson_matrix)
axis square
title('Pearson Matrix for 14 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Pearson_matrix_for_14_Subjects', '-dpng',  '-r600')

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

print('Absolute_Pearson_matrix_for_14_Subjects', '-dpng',  '-r600')

%% Compute Adjacency Matrix
Eye_Matrix = eye(44, 44);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
xlswrite('Adjacency_Matrix.xlsx', Adjacency_Matrix);

figure(4)
imagesc(Adjacency_Matrix)
axis square
title('Adjacency Matrix for 14 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Adjacency_Matrix_for_14_Subjects', '-dpng',  '-r600')

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





