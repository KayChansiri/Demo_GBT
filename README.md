# The Art of Gradient Boosting: A Practical Approach

Welcome to the third article in my series on ensemble techniques for machine learning. In my previous post, I discussed Random Forest, an ensemble technique that combines decision trees in parallel. In this post, I will introduce you to boosting, another powerful ensemble technique that works sequentially. For me, gradient boosting trees (GBT) always reminds me of building a sandcastle at the ocean when I was young. The way you build the first layer determines how the castle will look as you add each subsequent layer, all the way up to the roof. It’s like an artistic approach to machine learning, where each decision informs the next. Let’s take a closer look together today.

## What is Boosting?

**Understand Weak Learner:** Before diving into boosting, let me first introduce the concept of a weak learner. A weak learner, often represented by a single decision tree, can be a classifier or regressor whose predictive power is only slightly better than random guessing. Despite its simplicity and limited predictive performance, a weak learner becomes valuable when combined with other weak learners to create a more powerful model. It's also important to note that a weak learner is considered adequate as long as its predictions are not completely uncorrelated (perpendicular) with the observed values in a vector space. If they are, the weak learner is no longer useful.

**The Concept of Boosting:** Boosting is an ensemble technique where each classifier or regressor tries to correct the errors made by the previous one. Unlike bagging (used in Random Forest), which builds trees in parallel, boosting builds trees sequentially, creating more dependency between the trees in the model. 

**How Boosting Works:** You may wonder how boosting works. Unlike bagging, where some data samples (i.e., rows in your dataset) are reused while others might not be used at all (known as the 'out-of-bag process'), boosting uses all samples in the dataset to calculate errors for the first tree. The initial model is often called a weak learner because it is a simple model that may not perform well on its own. Initially, all samples contribute equally to the error calculation. After the first tree makes its predictions, the samples that were incorrectly predicted are given more focus. This updated focus is used to build the next tree, and this process continues sequentially, creating a series of weak learners that, together, form a strong learner. Boosting works well for problems with both high bias and variance, while bagging specifically works well for problems with high variance (i.e., overfitting).

Keep in mind that the concept of "weighting" is not applicable to all types of boosting. For GBT, which I discuss in this post, the focus is on adjusting residuals from one tree to the next until we achieve a tree with minimal errors, rather than adjusting weights. This weight adjustment approach is more prominent in Adaptive Boosting (AdaBoost), which I will cover in my next post.

**Feature Selection in Boosting:** Boosting generally uses all available features when training each weak learner. It does not randomly select a subset of features for each tree, unlike bagging in Random Forest. Instead, boosting focuses on fitting each tree to the residual errors made by the previous trees.

**Final Prediction in Boosting:**  The final prediction depends on what type of boosting we are talking about. For classification GBT, the final prediction calculates the sum of the log odds from the initial prediction and the weighted sum of the log odds from all trees. This cumulative sum is then converted to a probability using a sigmoid function to get the final prediction. I know that concept sounds confusing for now, but bare with me. I promise you will understand it better after you finish reading  the post. For regression, GBT calculate the final prediction as the sum of the initial prediction (e.g., the mean of the outcome) and the weighted sum of the residual corrections from all the trees. In this case, the weighted sum simply refers to the sum of residual corrections across the trees multiplied by the learning rate, which controls how much each tree's residual prediction contributes to the final prediction. Learning rates prevent overfitting by scaling down the updates. For AdaBoost, things work a bit differently. I will not focus on it now though.

For GBT boosting, we can write a one-line equation representing the final prediction process as below: 

We can write a one-line equation representing the boosting process as below: 


￼<img width="302" alt="F(x) = Fo(x) +" src="https://github.com/user-attachments/assets/d68bc611-51cf-4bd3-9ac1-98026ab9f3a0">

*F(x)* is the final boosted model.
*F<sub>0</sub>(x)* is the initial model (e.g., the mean for regression, the log of the odd for classification).
*h<sub>m</sub>(x)* is the weak learner at iteration 
*η* is the learning rate, which is a parameter that can be fine-tuned. 
*M* is the total number of iterations or trees.

## Data Preparation Assumptions for Boosting Techniques

1. **Handling of Scale**: Similar to bagging in Random Forest, boosting algorithms do not require standardization or normalization of the input features as the algorithm is based on decision trees, which are are invariant to the scale of the features. 
2. **Dimensionality:** Unlike certain algorithms that may suffer from the curse of dimensionality such as KNN, boosting can handle high-dimensional data relatively well. However, keep in mind that if the number of features is much larger than the number of samples, the model might suffer from overfitting issues, just like many ML algorithms. Dimensionality reduction techniques (like PCA, EFA) or feature selection based on your project or research objectives can help in such cases.
3. **Data Quality:** Boosting is sensitive to noisy data and outliers because it tries to correct errors from previous learners. If the data is noisy, boosting can overfit to the noise. Some preprocessing, like removing outliers or using regularization techniques, can help mitigate this issue.
4. **Handling Missing Values:** Many boosting algorithms like XGBoost, which I will discuss in my next post, can handle missing values internally. However, it is still a good practice to handle missing values before applying the model to improve model performance. Imputation techniques such as expectation maximization or multiple imputation are helpful if you have missing completely at nonrandom. For data suffering from missing at random and the missing percentage is low, a simpler technique such as single imputation, mean imputation for continuous predictors, or mode imputation for categorical predictors is helpful.
5. **Data Size:** Boosting techniques can be computationally expensive, especially with large datasets, because of their sequential nature. Certain boosting techniques such. as XGBoost or LightGBM provide optimizations for faster computations.


<img width="903" alt="Screen Shot 2024-08-28 at 1 48 22 PM" src="https://github.com/user-attachments/assets/524606fe-916e-4e7e-86b5-8d633beb57b2">

Figure 1. The Differences and Workflow of Boosting versus Bagging Algorithms 

## Types of GBT Boosting

GBT can be used with both classification and regression problems, unlike certain boosting algorithms such as AdaBoost that focuses on only classification problem. For GBT regression problem, the goal is to minimize a loss function, which measures the difference between the predicted values and the actual values. Common loss functions for regression include Mean Squared Error (MSE) and Mean Absolute Error (MAE). For classification problems, the goal is to minimize the log loss or cross-entropy loss. Minimizing log loss is equivalent to maximizing the log-likelihood, which you might have heard if you work with logistic regression before. Both minimizing log loss and maximizing log likelihood are mathematically related and represent the same objective. 

Let's spend time here for a sec to take a closer look at the concept of log loss. Bascially, log loss measures how well a classification model's predicted probabilities match the actual class labels. It is a measure of error or "loss" that quantifies the distance between the true class labels and the predicted probabilities. The log loss formula for binary classification is as below. A lower log loss means better probability estimates from the model.

![Screen Shot 2024-09-12 at 2 27 57 PM](https://github.com/user-attachments/assets/05f64b7d-59f7-4ce3-bd0d-e9a6bdcdaeab)

Now, for log-likelihood, the concept measures how likely it is that the observed data (actual labels) would occur given the predicted probabilities by the model. The higher the likelihood, the better the model fits the data. If you are not familiar with the concept. In classification tasks, we aim to maximize the log likelihood because a higher log likelihood indicates that the model is producing predictions that align well with the actual data. The equaiton of the log likelihood can be represented as below: 

![Screen Shot 2024-09-12 at 2 34 05 PM](https://github.com/user-attachments/assets/930a0c62-e966-4b05-9514-28049655a59b)

Notice that the log likelihood is just the negative of the log loss (without the averaging and negative sign). That's why I mentioned previously that the concepts of log loss and log likelihood are mathematically similar.

On a side note, don't get confused between the concepts of log loss and log of the odds. Those are totally diferent things. The log loss measures the distance between the true labels and the predicted probabilities. The log of the odds (AKA logit) are defined as the ratio of the probability of the event occurring to the probability of it not occurring.

Now that you understand the types of GBT and the relevant concepts, let’s start with GBT for regression.

### 1.GBT for Regression

* The fundamental of GBT for regression is gradient descent. If you are not familiar with the concept, know that a gradient is a vector that points in the direction of the steepest increase of the loss function. The magnitude of this vector indicates how steep the slope is. In optimization, which is the basis of ML algorithms, we are interested in moving in the opposite direction of the gradient (the direction of the steepest decrease) to minimize the loss. This is why we focus on gradient 'descent' or the 'negative gradient.'
  
* To better understand how GBT for regression works, let’s start with an initial standalone leaf as the most basic predicted value before we beginning to build a tree. Just like a typical linear regression model, the most basic thing that would us understand the outcome is the intercept. Like the concept of intercept, the initial leaf for GBT for regression is simply the average of the observed outcome across all samples in the dataset. For example, if you are building a model to predict customer satisfaction and the average satisfaction score across all customers is 3.4, the first predicted value or the initial leaf is 3.4. This values stands alone without being associated with any tree. For simplicity, imagine that your data looks like this:

<img width="643" alt="Screen Shot 2024-09-16 at 4 55 24 PM" src="https://github.com/user-attachments/assets/f05ececa-2155-47ea-b224-3cb1f577b43f">


* For each data point, if we subtract the predicted satisfaction (i.e., 3.4) from the actual satisfaction scores, you get the residuals. These residuals in the Residual column represent the errors of the current model across customers.
  
* Now that we get the initial leaf and the residuals for all participants, let's build the first tree.

> Note that unlike trees in Random Forest, the trees we build for GBT predict residuals NOT the observed data, which in this case is the actual customer satisfaction scores.

* However, similar to a typical decision tree, at each node in this first tree of the GBT, the algorithm evaluates splits using predictors in the dataset, which could be demographic variables or any variables important for your project, to determine which split best reduces the loss function of the residuals. **Again I said the residuals** NOT **the actual outcome scores**. In this context where the outcome is continuous, the loss function is measured by mean squared error (MSE).
  
* For example, suppose we're considering a split on 'Income.' This first tree evaluates different split points (e.g., "Income < $50,000" vs. "Income ≥ $50,000"). For each possible split, it calculates the MSE of the residuals for the data points falling into the left and right child nodes after the split. The split value that results in the lowest total MSE for the residuals is chosen.
  
* Still confused? Suppose the algorithm is considering a split on the Income variable, with a threshold of $50,000. This means we create two groups of samples: 1) Group 1 representing customers with Income < $50,000, and 2) Group 2 representing customers with Income ≥ $50,000. For each group, the algorithm computes the mean of the residuals. Again, I emphasize the mean of the residuals NOT the actual customer satisfaction scores themselves. For instance, if the residuals for Group 1 (Income < $50,000) are [0.6,−0.4,−0.4], the mean residual for this group would be - 0.07. Similarly, the algorithm computes the mean residual for Group 2 in the same way.
  
* The algorithm then calculated the averaged residuals for each group (e.g.,-0.07 for the Group with income < $50,000 in this example). For each group, The algorithm calculates the Mean Squared Error (MSE) between the actual residuals and the predicted residuals, using the equation below as an example for Group 1:

<img width="619" alt="Screen Shot 2024-08-29 at 4 50 31 PM" src="https://github.com/user-attachments/assets/6dd17a75-9367-4e38-b17b-c1b68c3735f4">


The total MSE for this split is a weighted sum of the MSEs for each group: 

<img width="466" alt="Screen Shot 2024-08-29 at 4 51 31 PM" src="https://github.com/user-attachments/assets/b400fb5a-84f3-4d65-bc7f-32497281cf5f">

Here, *n1*  and  *n2*  are the number of samples in Group 1 and Group 2, and  *n* is the total number of samples. The split that results in the lowest Total MSE is chosen as the best split for that node. 

* Now that we got the averaged residuals from tree 1, we will use the residuals to update the predicted value (i.e., customer satisfaction score) for each customer. Note that in the real world, you will likely have more than one node (i.e., more than the income variable) as the predictors. However, I will just build the first tree using only income as the predictor here for simplicity. To update the predicted value for each customer, you will just pass the data point down the tree according to the decision rules at each node, find the residual for that value, and sum it with the initial leaf value to obtain the new predicted value. 


![Corrected_Income_Residual_Decision_Tree](https://github.com/user-attachments/assets/ff841cde-bccc-4548-ac45-f7a70e962e75)


For example, for customer 1 in the table, their income is less than 50,000 and the averaged residual for them is -0.07. Thus, the new preidcted value is:

New Predicted Value = Initial Leaf + (Averaged Residual from Tree *n*) = 3.4 + −0.07 . Now we do this for every customer and you wil get the updated table below: 



<img width="777" alt="Screen Shot 2024-09-17 at 12 07 51 PM" src="https://github.com/user-attachments/assets/d8ee5d54-2e96-42d4-b60e-cbc5b4e590ae">


Now, you may notice that the updated predicted values for certain customers such as customer number 5 (the predicted satisfaction score = 3.33) are closed to the observed value (3). Isn't this excellent? 

The answer is no. We are facing an overfitting issue because Tree 1, as the model, learns too quickly to predict the observed values, leading to a poential lack of generalizability to new, unseen data. To fix this issue, we need to apply **a learning rate**, a hyperparameter that we can fine-tune and that ranges from 0 to 1. Lowering the learning rate can scale down the influence of each new tree, reducing the overfitting problem. The equation is as below: 

New Predicted Value = Initial Leaf + (learning rate x Averaged Residual from Tree *n*)

With a learning rate of 0.1, the updated predicted value for the first customer is  3.4 + 0.1(−0.07). We do this for every customer and get the updated table as below: 



<img width="750" alt="Screen Shot 2024-09-17 at 11 42 15 AM" src="https://github.com/user-attachments/assets/6d5571d4-6e40-4373-9b4c-cbff522172a0">

Now can see that the predicted values for tree 1 are less closer to the observed values yet still closer to the actual values than the initial predicted value (i.e., 3.4).

With a lower learning rate, the model takes smaller steps toward reducing the loss, requiring more trees to reach a similar level of accuracy. This slower, more controlled process allows the model to capture underlying patterns gradually without fitting to noise. I often found that using more trees with a smaller learning rate can result in a model that generalizes better than a few large steps

The concept I explained above is applicable to building all next trees in the boosting process. In other words, for each new tree, we calculate residuals based on the updated predicted values from the previous tree. Then we use these residuals to continue building trees, each time reducing the model's error further as an iteractive process. The final prediction for any customer is the sum of the initial leaf, adjusted by the weighted residuals from each tree:

**Final Predicted Value=Initial Leaf+(Learning Rate×Residuals from Tree 1)+(Learning Rate×Residuals from Tree 2)+…**

### 2. GBT for Classification



