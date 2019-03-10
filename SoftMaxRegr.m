%%The bellow functions are from the assignments in the site deeplearning.stanford.edu :

%%Cost function
function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
 % numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - learning rate/weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data


theta = reshape(theta, numClasses, inputSize);
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
thetagrad = zeros(numClasses, inputSize);
y = groundTruth;
m = numCases;
% note that if we subtract off after taking the exponent, as in the text, we get NaN
td = theta * data;
td = bsxfun(@minus, td, max(td));
temp = exp(td);
denominator = sum(temp);
p = bsxfun(@rdivide, temp, denominator);
cost = (-1/m) * sum(sum(y .* log(p))) + (lambda / 2) * sum(sum(theta .^2));
thetagrad = (-1/m) * (y - p) * data' + lambda * theta;
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

%%%==============================================================
%%%==============================================================


%% Compute the Gradient descent
function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
 
eps = 1e-4;
n = size(numgrad);
I = eye(n, n);
for i = 1:size(numgrad)
    eps_vec = I(:,i) * eps;
    numgrad(i) = (J(theta + eps_vec) - J(theta - eps_vec)) / (2 * eps);
end
end
%%%==============================================================
%%%==============================================================

%% Compute the Gradient descent
function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)

% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for
 
if ~exist('options', 'var')
    options = struct;
end
 
if ~isfield(options, 'maxIter')
    options.MaxIter = 400;
end
 
% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);
 
% Use minFunc to minimize the function
addpath ../common/fminlbfgs
options.Method = 'lbfgs'; 
% Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
 % function value and the gradient. In our problem,
 % softmaxCost.m satisfies this
options.Display = 'iter';
options.GradObj = 'on';
 
[softmaxOptTheta, cost] = fminlbfgs( @(p) softmaxCost(p, numClasses, inputSize, lambda,  inputData, labels), theta, options);
 
% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
end                          

%%%==============================================================
%%%==============================================================

%%  Prediction of the model
function [pred] = softmaxPredict(softmaxModel, data)
 
% data - the N x M input matrix, where each column data(:, i) corresponds to a single test set
% produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));
 
size(pred); %   1 10000
size(data); % 784 10000
size(theta);%  10     8
 
p = theta*data;
[junk, idx] = max(p, [], 1);
 
pred = idx;
 
end



%%%==============================================================
%%%==============================================================

%%  Softmax Regression
 
%% Initialise constants and parameters
inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)
lambda =  1e-4; % Weight decay parameter - learning rate
 
%%  Load data
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
 
inputData = images;
 
% For debugging purposes,  the size of the input data has been reduced in order to speed up gradient checking. 
DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
    inputSize = 8;
    inputData = randn(8, 100);
    labels = randi(10, 100, 1);
end
% Randomly initialise theta (weights)
theta = 0.005 * randn(numClasses * inputSize, 1);
%  Implement softmaxCost in softmaxCost.m. 
[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
                                     
%%  Gradient checking

if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, inputSize, lambda, inputData, labels), theta);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 
 
    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
end
 
%% Learning parameters
% training softmax regression code using softmaxTrain  (which uses minFunc).
 
options.MaxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);
 
%%  Testing the test imagies 
%    softmaxPredict.m which return predictions given a softmax model and the input data.
 
images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
 
inputData = images;
 
% implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
 
% Accuracy is the proportion of correctly classified images
acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);     

The following code was implemented as a part of the assignments in the course Neural Networks in Aristotle University of Thessaloniki:

% load training set and testing set
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');
 
tic;   %start  timer
 
% parameter setting
alpha = 0.1; % learning rate
beta = 0.01; % scaling factor for sigmoid function
train_size = size(train_set);
N = train_size(1); % number of training samples
D = train_size(2); % dimension of feature vector, 784pixels
n_hidden = 300; % number of hidden layer units
K = 10; % number of output layer units
 
% initialize all weights between -1 and 1
W1 = 2*rand(1+D, n_hidden)-1;  
W2 = 2*rand(1+n_hidden, K)-1;  
epochs = 200;  
Y = eye(K);   % output  
test_size = size(test_set);
 
% training 
for i=1:epochs
    disp([num2str(i), ' epoch']);  % display the number of the epoch
    permutation = randperm(N);  %randomize the order of j
    for j=1:N
        index = permutation(j);  %new index after permutation
        % propagate forward 
        input_x = [1; train_set(index, :)'];
        hidden_output = [1;sigmf(W1'*input_x, [beta 0])];
        output = sigmf(W2'*hidden_output, [beta 0]);
        % back propagation
        % compute the error of output unit c
        outlayer_delta = (output-Y(:,train_label(index)+1)).*output.*(1-output);
        % compute the error of hidden unit h
        hiddenlayer_delta = (W2*outlayer_delta).*(hidden_output).*(1-hidden_output);
        hiddenlayer_delta = hiddenlayer_delta(2:end);
        % update the weights
        W1 = W1 - alpha*(input_x*hiddenlayer_delta');
        W2 = W2 - alpha*(hidden_output*outlayer_delta');
    end
    
    toc; %end  timer 
	
    %train testing
    success = 0;
    for j=1:test_size(1)
    input_x = [1; test_set(j,:)'];
    hidden_output = [1; sigmf(W1'*input_x, [beta 0])];
    output = sigmf(W2'*hidden_output, [beta 0]);
    [max_unit,max_idx] = max(output);
         if(max_idx == test_label(j)+1)
             success =success + 1;
         end
    end
    perc_succ = success/test_size(1);
    perc1=100*perc_succ;
    disp([num2str(perc1), '% '])
end

logit=zeros(test_size(1),10);

% testing 
num_correct = 0;
for i=1:test_size(1)
    input_x = [1; test_set(i,:)'];
    hidden_output = [1; sigmf(W1'*input_x, [beta 0])];
    output = sigmf(W2'*hidden_output, [beta 0]); 
    logit(i,:) = softmax(output'); 
    [max_unit,max_idx] = max(output);
    if(max_idx == test_label(i)+1)
        num_correct = num_correct + 1;
    end
end

% computing accuracy
accuracy = num_correct/test_size(1);
perc=100*accuracy;
disp(['Total accuracy ', num2str(perc), '%'])
