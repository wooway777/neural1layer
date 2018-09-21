%Problem Set 2 Qurstion 3
%number of nodes
d1s = [1 5 10 15 25 50];
%step size
eta = 0.1;
%number of iterations
N = 5000;
%activation function
activation = "sigmoid";
%task
task = "classification";

train = importdata("Problems-1-2-3/Spam-Dataset/train.txt");
test = importdata("Problems-1-2-3/Spam-Dataset/test.txt");

%train size
[m, d] = size(train);
d = d - 1;
%test size
[m_t, d_t] = size(test);
d_t = d_t - 1;

%data and labels
train_data = train(1:m, 1:d);
train_label = train(1:m, d+1);
test_data = test(1:m_t, 1:d_t);
test_label = test(1:m_t, d_t+1);

train_imports = ["Problems-1-2-3/Spam-Dataset/CrossValidation/Fold1/cv-train.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold2/cv-train.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold3/cv-train.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold4/cv-train.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold5/cv-train.txt"];
test_imports = ["Problems-1-2-3/Spam-Dataset/CrossValidation/Fold1/cv-test.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold2/cv-test.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold3/cv-test.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold4/cv-test.txt" ...
    "Problems-1-2-3/Spam-Dataset/CrossValidation/Fold5/cv-test.txt"];

base = "Problems-1-2-3/Spam-Dataset/setting-files/";

cv_size = length(train_imports);
cv_errors = zeros(1, length(d1s));
training_errors = ones(1, length(d1s));
test_errors = ones(1, length(d1s));

for index = 1:length(d1s)
    d1 = d1s(index);
    load(base + "w1_" + d1 + ".mat")
    load(base + "b1_" + d1 + ".mat")
    load(base + "w2_" + d1 + ".mat")
    load(base + "b2_" + d1 + ".mat")
    %train and test
    X = train_data;
    Y = train_label;
    
    adjusted_Y = (Y + 1)/2;
    
    %train
    for n = 1:N
        [w_1, b1, w_2, b2] = update_wb(X, adjusted_Y, w_1, w_2, b1, b2, eta, activation, task);
    end

    eta_hat = g(fwb(X, w_1, w_2, b1, b2, activation), activation);
    predictions = sign(eta_hat - 1/2);
    training_errors(index) = classification_error(predictions', Y);
    eta_hat_t = g(fwb(test_data, w_1, w_2, b1, b2, activation), activation);
    predictions_t = sign(eta_hat_t - 1/2);
    test_errors(index) = classification_error(predictions_t', test_label);
    
    %cross validation
    for fold = 1:cv_size
        cv_train = importdata(train_imports(fold));
        cv_test = importdata(test_imports(fold));

        %train size
        [cv_m, cv_d] = size(cv_train);
        cv_d = cv_d - 1;
        %test size
        [cv_m_t, cv_d_t] = size(cv_test);
        cv_d_t = cv_d_t - 1;

        %data and labels
        cv_train_data = cv_train(1:cv_m, 1:cv_d);
        cv_train_label = cv_train(1:cv_m, cv_d+1);
        cv_test_data = cv_test(1:cv_m_t, 1:cv_d_t);
        cv_test_label = cv_test(1:cv_m_t, cv_d_t+1);

        %make it clear for error checking
        cv_X = cv_train_data;
        cv_Y = cv_train_label;

        load(base + "w1_" + d1 + ".mat")
        load(base + "b1_" + d1 + ".mat")
        load(base + "w2_" + d1 + ".mat")
        load(base + "b2_" + d1 + ".mat")

        adjusted_cv_Y = (cv_Y + 1)/2;
        
        %train tmd
        for n = 1:N
            [w_1, b1, w_2, b2] = update_wb(cv_X, adjusted_cv_Y, w_1, w_2, b1, b2, eta, activation, task);
        end

        eta_hat = g(fwb(cv_X, w_1, w_2, b1, b2, activation), activation);
        predictions = sign(eta_hat - 1/2);
        cv_errors(index) = cv_errors(index) + classification_error(predictions', cv_Y)/cv_size;
    end
end

figure

xlabels = [1:6];

plot(xlabels, training_errors, '.-', xlabels, test_errors, '.-', ...
    xlabels, cv_errors, '.-', 'MarkerSize', 15, 'LineWidth', 2)

title('Error Curves')
xlabel('d1 = [1 5 10 15 25 50]')
ylabel('Error')
lngd = legend('Training Error','Test Error', 'Cross Validation Error');
set(lngd, 'Location', 'NorthEast')
set(lngd, 'fontsize', 10)

selected_index = find(cv_errors == min(cv_errors));
if length(selected_index) > 1
    selected_index = selected_index(1);
end
disp("d1")
d1s(selected_index)
disp("Training Error")
training_errors(selected_index)
disp("Test Error")
test_errors(selected_index)
disp("CV Error")
cv_errors(selected_index)
