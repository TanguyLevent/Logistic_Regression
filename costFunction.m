function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

H = sigmoid(X*theta);

J = - (1/m)*sum((y'* log(H) + (1 - y)'*log(1-H)));
grad = (1/m)*(H-y)'*X;


end
