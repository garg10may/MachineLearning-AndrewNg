function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

#{
%1st Method, programitcally
v = 0;
for i = 1:m
  a = X(:,2); % only second column is needed, first column contains ones
  v = v + (theta(1) + theta(2)*a(i) - y(i))^2;
  % above is simple formula for J(theta)
endfor

J = v/(2*m);
#}

%2nd Method, Using matrix math
sqrErrors = ( X*theta - y).^2;
J = 1/(2*m) * sum(sqrErrors); # Just one line :)

end
