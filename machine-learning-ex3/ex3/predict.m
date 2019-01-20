function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % number of rows in X
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ ones(m,1) X];

% looping through and filling prediction for each example
#{
for i = 1:m %Just appending a row of ones 
  temp = X'(:,i); % Just taking one example at a time
  layer2_nodes = sigmoid(Theta1 * temp); %Take X all rows but only ith column
  layer2_nodes = [1 layer2_nodes']; % adding fake 1st node for hidden layer
  h = sigmoid(Theta2 * layer2_nodes');
  [not_needed, p(i)] = max(h); %this gives the index of the max value
endfor
#}

% using fully vectorized approach 
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
htheta = sigmoid(a2 * Theta2');

[temp, p] = max(htheta, [], 2);


% =========================================================================
end
