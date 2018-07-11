# neural_network_i_matlab

## Multi-class Classification (One-vs-all)

### 1. Dataset
![Dataset](https://raw.githubusercontent.com/guoqi228/neural_network_i_matlab/master/fig_1_dataset.png)

### 2. Logistic regression vectorized cost funtion and gradient
```
theta_zero = [0; theta(2:end)];
J = mean((-y).* log(sigmoid(X * theta)) - (1 - y).* log(1 - sigmoid(X * theta)))...
    + lambda/2/m * sum(theta_zero.^2);
grad = (X' * (sigmoid(X * theta) - y))/m + lambda/m*theta_zero;
```

### 3. One-vs-all
```
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for i = 1:num_labels
      [theta] =  fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
          initial_theta, options);
      all_theta(i,:) = theta';
  end
end
```

### 4. Prediction accuracy
```
function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  p = zeros(size(X, 1), 1);
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  for i = 1:m
      input = X(i,:)';
      out = sigmoid(all_theta * input);
      [M,I] = max(out);
      p(i,1) = I;
  end
end
```

Prediction accuracy = 94.9%

## Neural Network

### 1. Model
![Model](https://raw.githubusercontent.com/guoqi228/neural_network_i_matlab/master/fig_4_model.png)

### 2. Feedforward propagation and prediction
```
function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  for i = 1:m
    input = [1 X(i,:)]';
    a2 = sigmoid(Theta1 * input);
    a2 = [1; a2];
    output = sigmoid(Theta2 * a2);
    [M,I] = max(output);
    p(i,1) = I;
  end
end
```
prediction accuracy = 97.5%
