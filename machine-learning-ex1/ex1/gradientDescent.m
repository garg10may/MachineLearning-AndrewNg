function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% Method 1, programatically
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    v1=0;
    v2=0;
    a = X(:,2); % only second column needed, it has the values
    
    % Caclulate for theta 0
    for i = 1:m
      v1 = v1+  (theta(1) + theta(2)*a(i) - y(i));   
    endfor
    
    % calculate for theta 1
    for i = 1:m
      v2 = v2+ ((theta(1) + theta(2)*a(i) - y(i))*a(i)); 
    endfor 
    
    temp0 = theta(1) - alpha * (1/m) * v1;
    temp1 = theta(2) - alpha * (1/m) * v2;
    
    % update theta after calculating both 
    theta(1) = temp0;
    theta(2) = temp1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

%Method 2, using Vector Math, but since here the output also needs J history 
% this would fail submission, otherwise it calculates theta fine
%theta = pinv(X' * X) * X' * y;
