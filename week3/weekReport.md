The Plack Low can not fit with linear regression.    
It is a non liear process. The problem can not be transform into format of y = WX.  
I tried with logarithm. The equation is still a non-linear process.  
So it might be solved with other modle, but not with linear-regression.  
  
As well, for very larg value. Must pay attention to if the scale of varient exceed.  

Day3
Self build a linear regression model.  
The input data are generate with y_ref = X @ w_ref + b + GaussianWhiteNoise  
The expect value of the loss after training should be sigma ** 2  

The gradient of each weight is the inner product between its corresponding feature vector and the prediction residual.  
Expanding the residual shows that the gradient depends on the inner products between that feature and all other feature vectors,  
which are exactly the entries of XTX.  
Therefore, feature correlations determine how the gradients of different weights are coupled during optimization.

## Gradient Interpretation in Linear Regression

For the weight (w_j), the gradient of the mean squared error loss is

$$
\frac{\partial L}{\partial w_j}=\sum_{k=1}^{m}w_k\left(x_j^{T}x_k\right)
x_j^{T}y.
$$

This expression provides an important interpretation of gradient descent in linear regression.

$(x_j^{T}x_j)$ measures how strongly feature (j) contributes to the gradient of its own weight.
$(x_j^{T}x_k)$ measures the interaction (or correlation) between feature (j) and feature (k). Consequently, it determines how the weight (w_k) influences the gradient of (w_j).

Therefore, the gradient of each weight is not determined solely by its corresponding feature. Instead, it is coupled with the gradients of all other weights through the pairwise inner products between feature vectors, which are precisely the entries of the Gram matrix,

$X^{T}X.$


Equivalently, the gradient of the entire weight vector can be written as
$$
\nabla L =  X^{T}Xw - X^{T}y.
$$
This equation clearly separates the two roles:

* $(X^{T}X)$ describes the relationships between features and determines the curvature of the loss surface.
* $(X^{T}y)$ describes the relationship between the features and the target values, indicating the direction in which the model should move to reduce the loss.

As a result, **the correlation between features directly affects both the gradient and the optimization process**. When two features are highly correlated, the corresponding off-diagonal entries of (X^{T}X) become large, causing the gradients of their weights to become strongly coupled. Geometrically, the quadratic loss surface becomes an elongated valley. Consequently, gradient descent converges more slowly and the estimated weights become less stable, even though the model may still achieve accurate predictions.

The KEY CONCEPT is: THE WRIGHT ARE NOT ISOLATED. EACH WEIGHT NEED CONSIDER CORRLEATION FROM OTHER WEIGHTS. WHICH MEANS CONSIDER CORRLEATION OF THE FEATURES.

DAY4

When can you solve the problem of polynomial regression exactly?  
The problem must could be transformed into a linear relationship.  
The colsed_form solution of wright exist, as input matrix has no column dependency:
        $$    | a_1 x_1^2 x_1^3 x_1^4 ... x_1^n | $$
        $$    | a_2 x_2^2 x_2^3 x_2^4 ... x_2^n | $$                                                          
        $$    | ...                             | $$
        $$    | a_m x_m^2 x_m^3 x_m^4 ... x_m^n | $$


Give at least five examples where dependent random variables make treating the problem as IID data inadvisable.  
If the input sample need dependency on other dependency the iiD will break.  
All time-series samples. 
Or thh sample it self has significant classify.  
For exapmle, date from several users, the samples from same user will have some pattern different from other users.  
Which will braek the idenpendent.
Another example is when the samples themselves belong to distinct groups.   
For example, if a dataset contains data collected from several users, the samples from the same user are likely to share common characteristics,   
making them more similar to each other than to samples from other users. As a result, the samples are not independent, violating the IID assumption.  

Can you ever expect to see zero training error? Under which circumstances would you see zero generalization error?
For zero training error, we need the input data to be noise free.  
$ y_{predict}=XW + b +  GaussianWhiteNoise$
The expectation of train noise is:  
$E[train_loss]=sigma**2$  
