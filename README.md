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

To understand gradient boosting for classification, we need to first grasp the concepts of log odds (logit), odds, probability, and log likelihood. These are foundational concepts of logistic regression, which I explained in a previous post.

Let’s breifly discuss these concepts. For binary outcomes (e.g., yes/no, true/false), a linear algorithm cannot be directly used to predict the outcome because the predictions might fall outside the intended boundary [0,1]. To restrict the predicted values within this range, we use functions like the sigmoid function. The sigmoid function is mathematically represented as below (note that *z* is the output of a linear regression equation, which can be represented as *B*<sub>*0* + *B*<sub>*1*</sub>*X*<sub>*1*</sub> + *B*<sub>*2*</sub>*X*<sub>*2*</sub> + ...

<img width="252" alt="Screen Shot 2024-09-01 at 11 58 29 AM" src="https://github.com/user-attachments/assets/039a8314-f0c0-42cd-a46c-b5db8002d414">

In the sigmoid function, a typical linear regression model is transformed such that its output values are constrained within the  [0,1] range. This transformation allows the output to be interpreted as a probability, as shown in the plot below.

<img width="758" alt="Sigmoid Function" src="https://github.com/user-attachments/assets/47a8cd51-f8ef-4c50-b27b-4c949ab18c42">

However, it is challenging to interpret this non-linear plot directly. For instance, you cannot determine exactly how a one-unit increase in the predictor results in a one-unit increase in *Y* because the relationship is not linear. Therefore, we apply the log function to transform the non-linear sigmoid output back into a linear form. This transformation is expressed as:

<img width="249" alt="Logit(p)" src="https://github.com/user-attachments/assets/02ecc822-0ebc-4580-a50f-06aec399466f">

In the equation,*p* is the probability obtained from the sigmoid function. After obtaining a linear function with log odds, it may still not be intuitive to interpret how a one-unit increase in *X* results in an increase in *Y* in terms of log odds. Therefore, we often convert log odds to an odds ratio, which essentially refers to the ratio of the probability of an event of interest happening to the probability of it not happening, given a one-unit increase in predictor *X*. Probability, log odds (logit), and odds ratios can be converted back and forth using the following equations:

![Screen Shot 2024-09-17 at 12 23 39 PM](https://github.com/user-attachments/assets/a388fb2d-ebdd-461f-9b18-f76e80f4adc5)

Now that you understand these concepts, let’s return to gradient boosting for classification.

Unlike gradient boosting for regression, where the initial leaf is the average of the outcome, the initial leaf of gradient boosting for classification is the log odds (i.e., logit), which is equal to the logarithm of the probability of an event happening versus not happening, as explained earlier.
Consider the same example of age and income predicting customer satisfaction, but this time the outcome is whether the customer is satisfied (1) or dissatisfied (0). Imagine we have 5 customers, and two of them are satisfied, as depicted in the table below:

<img width="352" alt="Screen Shot 2024-09-17 at 12 38 53 PM" src="https://github.com/user-attachments/assets/d503aa2c-4c7d-4178-9c74-f62f7e4a6db1">



The log of the odds can be calculated using the equation:

<img width="553" alt="Log Odds (Logit)" src="https://github.com/user-attachments/assets/41e6f753-20e9-4a06-9d93-de1670cf8401">

In our example, the log of the odds is:

<img width="481" alt="Log Odds (Logit) = log" src="https://github.com/user-attachments/assets/4a973420-821e-4ff4-be3a-203f3ee8ca53">

This logit value is our initial leaf. Now, you may wonder how we use the log odds to get the residual values so that we can use them to grow a tree when the values in our table are just 0 and 1. The answer is that we convert the log odds to a probability so that we can subtract the predicted probability from the actual observed values (0 or 1).
If you recall the equations above, we can get the probability as:


<img width="210" alt="1 + e0 4" src="https://github.com/user-attachments/assets/2f3d6351-fd79-4f33-b9df-6e0a56ebfe0f">

Now that we have the probability as our initial leaf, we can calculate the residuals by subtracting the predicted probability from the actual observed values (0 or 1), just like we did in gradient boosting regression.


<img width="676" alt="Screen Shot 2024-09-17 at 12 38 10 PM" src="https://github.com/user-attachments/assets/6acebd97-40ed-41d6-a7fb-cdce6a18bc5c">

Similar to what we did earlier in gradient boosting for regression, these residuals will then be used to build our first tree using the same logic we used prevoiusly for GBT for regression. For example, if we want to use income to predict customer satisfaction, the algorithm evaluates various potential cut-off points (e.g., "income < 40,000", "income < 60,000", etc.) and selects the one that minimizes the Gini Index or Cross-Entropy, making the leaf nodes as "pure" as possible in terms of residuals. If "income < 50,000" is considered a potential cut-off point, the residuals of customers with an income below this threshold are considered in the split calculation. The goal is to find the split that results in more homogeneity in terms of their residuals. We can calculate this using the formula below: 


<img width="501" alt="Gini = 1 - (probability of each class)2" src="https://github.com/user-attachments/assets/22b5583f-3742-44cc-9369-757ae888c356">

If the split results in nodes where the residuals are closer to zero, it is considered a good split. The process involves iteratively adjusting the cut-off point until the leaf nodes are as pure as possible, meaning they contain residuals that are minimal and represent better predictions for the classification task. Let’s imagine the first tree we obtained is shown below:

![output-6](https://github.com/user-attachments/assets/36830b57-65d2-4c71-b35f-4de15c7d882c)

Note that for GBT tasks, residuals are referred to as "pseudo-residuals" because they represent the difference between the actual outcomes and the predicted residuals (i.e., averaged residuals for a regression task and probabilities for a classification task), rather than the difference between the actual and predicted values as in linear regression. A more technical way to explain this is that pseudo-residuals are derived from the gradient of the loss function used, such as the negative log-likelihood (log loss) or binary cross-entropy for classification tasks, and Mean Squared Error (MSE) for regression tasks.

Now that we have the residuals as the final leaves of the tree and the initial leaf (e.g., −0.4), we can calculate the predicted probability for each customer.
However, since the initial leaf is in the log of the odds (logit) form and the tree is built based on probabilities, we can't simply sum the log odds and residuals. As explained earlier, log odds and probabilities are two different representations.

To address this issue, we need to perform some mathematical transformations. For each final leaf, we will plug its residual into the formula below to get the adjusted residuals in a form that allows us to combine the value with the initial leaf.

<img width="671" alt="Adjusted Residual" src="https://github.com/user-attachments/assets/d49ea336-2261-47ff-a046-3a32b2adc817">

Let's use Final Leaf 1 as an example and apply the formula to calculate the adjusted residual. Note that the previous probability for all customers in this leaf is 0.4 (see Table 2). Thus, for the numerator, we have 0.6+(−0.4)+(−0.4)=0.6−0.8 = −0.2. For the denominator, we calculate 0.4×0.6+0.4×0.6+0.4×0.6=0.72. Therefore, the adjusted residual for the first leaf is −0.2/0.72=−0.28. If you repeat this calculation for all final leaves, you will get a tree that looks like the one shown below:

![output-7](https://github.com/user-attachments/assets/4272a8a3-47b1-4827-88dd-236286e7f699)

Now, we have the final output for each leaf in a form that can be combined with the initial leaf, which is in the log-odds form. To do this, we can use the formula below.

<img width="649" alt="New Log Odds - Initial Log Odds + (Learning Rate × Adjusted Residual)" src="https://github.com/user-attachments/assets/d6d3347b-185d-4220-96ee-648fae5cb0d4">

For Customer 1, we calculate −0.4+(0.1×−0.28). Note that the learning rate can be fine-tuned, but I will use 0.1 for now just for demonstration purposes. We repeat this calculation for every customer to obtain the log-odds for each of them. Then, we convert the log-odds to probabilities so that we can subtract these probabilities from the previous probabilities in the table to get the new set of residuals, as shown in the table below:


<img width="864" alt="Screen Shot 2024-09-17 at 1 00 19 PM" src="https://github.com/user-attachments/assets/e14359b9-c935-4450-b365-e7a5c6e4b49b">

You may notice that the new residuals for some customers are smaller than their initial residuals. This indicates that we are on the right track. However, for certain customers, the new residuals are larger than the original ones. This is why we need to keep repeating the process until all residuals become sufficiently small or until we reach a maximum number of trees specified, indicating that further improvement is minimal. By controlling the learning rate and the number of trees, we can ensure that the model does not overfit the data and generalizes well to unseen data. The ultimate goal is to build a robust model that balances bias and variance. In the end, we will have *n* trees, where *n* is a parameter we fine-tune. 

When we get a new, unseen data point, we will run it through every tree and then sum the final log-odds predictions **across all trees**, along with the initial leaf (-0.4), to obtain the final predicted log-odds for that specific data point. We then convert the log-odds into a probability to obtain the final prediction. Typically, we set the threshold at 0.5, meaning that if someone has a final probability greater than 0.5, their customer satisfaction will be coded as 1. If it is less than 0.5, it will be coded as 0.

## Parameters to Fine-Tune in Gradient Boosted Trees 

Now that you understand the concepts of how GBT for regression and classification work, let's talk more about which paramaters can be fine-tuned.

1. **Learning Rate (Range between 0 and 1):**  

GBTs include a learning rate (also known as shrinkage or step size) that scales the contribution of each tree. A smaller learning rate (closer to 0) requires more trees to model the data but often results in a more robust model that generalizes better to unseen data. A larger learning rate (closer to 1) may speed up the learning process but risks overfitting if the model captures too much noise from the training data. The choice of learning rate is a trade-off between the number of trees needed and the model's generalization ability.

2. **Tree Depth:**  

The individual trees in GBTs are usually shallow (typically not more than 40 leaves from what I have seen), ranging from 1 to a few levels deep. These shallow trees, again known as "weak learners," are not very powerful on their own but become highly effective when combined through the boosting process. Deeper trees provide more complex decision boundaries, leading to faster learning but also increasing the risk of overfitting, which adds more variance to the model. Conversely, shallower trees take smaller steps in learning (learning more slowly), requiring more iterations to capture the patterns in the data. However, they tend to reduce variance, leading to a more stable model. The optimal depth depends on the complexity of the underlying data.

3. **Regularization Techniques:**  
In addition to the learning rate, GBTs can be regularized through other techniques, such as subsampling the training data (also known as stochastic gradient boosting) and controlling the complexity of the trees (e.g., limiting tree depth, minimum samples per leaf). These techniques help prevent overfitting to the training data, especially when dealing with large datasets.

## Example 

Now that you have a strong foundation of what GBT is, let's take a look at the real world example. I use this dataset from Kaggle: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data. This dataset consists of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak — ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. I recommend you to go to the link and read more in detail about what is the dataset about. Here, we will use all of the predictors for a regression task. The target column is num column consisting of 5 levels including 0 = no heart disease, 1 = mild heart disease, 2 = moderate heart disease, 3 = severe heart disease, and 4 = critical heart disease. The dataset is not large as we have 16 x 920. 

There are many libraries that can be used to perform GBT for regression. I will use XGBoost today because of the following reasons. First, the algorithm is is known for its speed, efficiency, and performance. Second, XGBoost can automatically handle missing values by learning the optimal path for missing data during training. Thus, we don't have to bother dealing with missing values, which are present in certain columns of the dataset, although I encourage you to deal with missing values first to increase the model performance if you have some time. Third,  XGBoost supports parallel processing, making it faster compared to traditional gradient boosting methods. Finally, the algorithm supports both classification and regression tasks. I will perform both of them in this demo.

There are other libraries out there available for GBT such as LightGBM, developed by Microsoft, which is designed to be highly efficient, especially for large datasets.  Another interesgting libray is CatBoost, which particularly useful for datasets with categorical features as we do not need to do any extensive preprocessing (like one-hot encoding). The library also provides strong regularization to avoid overfitting. Since our dataset today is small and does not feature much of categorial variables, I opted for XGBoost.

## Data Preparation 
 Before we get started, we have to make sure first that the dataset meets of all the assumptions needed by GBT, although not many are there compared to other algorithms like linear regressions. 

**1. Data Distribution**
GBT does not require any specific assumptions about the distribution of the data (e.g., normality, homoscedasticity, linearity). GBT models can capture complex, non-linear relationships between features and the target variable, making them highly flexible and robust for a wide range of data types. Thus, there is no need to prepare the data for data distribution assumption.  There is also no need to check for  multicollinearity as GBT models are decision tree-based, meaning they split data based on feature values rather than fitting coefficients to predictors as in linear regression. Tree-based models are generally robust to multicollinearity because they do not assume independence among predictors. Instead, they select the most important features at each split, reducing the impact of correlated features. Thus, if two features are highly correlated, the model will likely choose one over the other based on how much they reduce the loss function (e.g., Mean Squared Error for regression tasks).

However, note that  independence of observations is still assumption that needs to be met in GBT like other machine learning models, althiugh the algorithm can handle highly correlated features. If there is dependency among observations (e.g., time series data), additional steps such as feature engineering or adding lag variables are necessary. This is not for our current dataset, which is cross-sectional.

**2. Handling of Feature Types**
For the library that we use today like XGBoost, categorical variables need to be encoded (e.g., one-hot encoding, label encoding) before training. We have the 'dataset' variable, which indicates location of data collection (3 levels), and 'cp', which indicates chest pain type with three levels. So, we will start with one-hot encode those variable first. 

```ruby
data = pd.get_dummies(data, columns=['dataset', 'cp']) 
```

**3. Handling of Missing Values** 
While GBT algorithms like XGBoost can handle missing values internally as I mentioned earliier, it is generally good practice to deal with missing values as part of the data preprocessing step. As the two columns that have missing values (i.e., trestbps and chol) are all continuous and the missing percentages are not that high, I will use  a mean imputation technique. For 'fbs', which is categorial, I will use mode for imputation. 

```ruby

# Perform mean imputation for the columns 'trestbps', 'chol', and 'fbs'
data['trestbps'].fillna(data['trestbps'].mean(), inplace=True)
data['chol'].fillna(data['chol'].mean(), inplace=True)

# Calculate the mode of the 'fbs' column
fbs_mode = data['fbs'].mode()[0]

# Impute missing values with the mode
data['fbs'].fillna(fbs_mode, inplace=True)


```

**4. Handling of Outliers**
GBT algorithms are relatively robust to outliers, especially compared to linear models. However, extreme outliers can still impact model performance by creating overly specific splits. Note that when I talked about outliers, the focus is generally on continuous predictors, not categorical predictors. To check for outliers, I will use  the Interquartile Range (IQR). Outliers are values that lie below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, where Q1 is the first quartile and Q3 is the third quartile. Then I will use boxplotsto visualize the outliers.


```ruby

import matplotlib.pyplot as plt

# List of columns to check for outliers
columns_to_check = ['age', 'trestbps', 'chol', 'thalch', 'num']

# Use descriptive statistics (IQR) to identify potential outliers
for column in columns_to_check:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    # Print outliers information
    print(f"Column: {column}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outliers:\n{outliers[column].values}")
    print("-----------------------------------------------------")

# Visualize the data using boxplots for continuous columns
for column in columns_to_check:
    plt.figure(figsize=(10, 4))
    plt.boxplot(data[column].dropna(), vert=False)
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()

```

The output indicate zero outliers for age, 28 for trestbps,  185 for chol, 2 for thalch, and 0 for num. The one with the highest outlier is chol. The rest is neglible. I decided to not do anything with the outliers of cholesterol level as the data realistically represent people at risk of the heart diseases -- a number of them woukd have extremely high choresterol level (see the box plot below). Thus, you have to asnwer yourself if you need to do anything for outliers in your own project, using previous litrature or the project objective as the guidances. 



![Screenshot 2024-09-17 at 4 22 22 PM](https://github.com/user-attachments/assets/3a76a2d2-b95a-4dbc-aa56-9bdf332d0599)

**5. Sufficient Sample Size**
While GBTs can perform well with smaller datasets, having a sufficient sample size is important to build robust models. Extremely small sample sizes can lead to overfitting, while large datasets generally benefit from the boosting process to improve model accuracy. In our case, the data set is on the smaller side. To prevent overfitting, I will deal limit and fine-tune some paramaters as you will see later.

**6. Feature Scaling** 
Unlike other algorithms such as KNN, GBT models do not require feature scaling (e.g., standardization, normalization) because they are based on decision trees, which split based on feature values rather than their magnitudes. However, scaling may still be useful in cases where features have vastly different ranges and could affect interpretability or the influence of certain features in the model.

**7. Feature Engineering**

Like other algorithms, you may create new features that capture non-linear relationships or interactions between variables. This can further enhance the predictive power of GBT models. These newly created variables should be also informed by previous lietrature or rigurous concepts. Since I am not a researcher in heart disease, I will ignore the procees  and focus on only existing varibales for now.

**8. Class Imbalance**

If you work with GBT for classificaiton tasks, checking if you are dealing with imbalanced classes in the ouctome is critical. If you encounter the issue, consider using techniques like SMOTE (Synthetic Minority Over-sampling Technique), undersampling, or adjusting class weights to improve model performance. I have explained more about what class imbalance is anmd how to deal with the situatation in this post. 

Class imbalance for preddictors is not typically a concern. Features can have varying distributions, and that's generally acceptable. The focus should be on whether these features are useful and predictive of the outcome variable.

If there are categorical predictors with rare categories (e.g., you have many White and Black for the race variable more than Asian and Native Inidians), it might not be considered a class imbalance issue but rather a data sparsity issue. You can handle this by combining rare categories together or using specialized encoding techniques like target encoding.

**9. Data Type**
XGBoost and most machine learning algorithms require numerical input data. For this specific dataset, after the one-hot encoding function, several variables are coded as boolen. I will convert those binary boolean variables to numerical binary (e.g., 0 and 1) before using them in XGBoost.
```ruby

# List of boolean columns to convert to numerical binary
boolean_columns = [
    'dataset_Cleveland', 
    'dataset_Hungary', 
    'dataset_Switzerland', 
    'dataset_VA Long Beach',
    'cp_asymptomatic', 
    'cp_atypical angina', 
    'cp_non-anginal', 
    'cp_typical angina',
    'restecg_lv hypertrophy',        
    'restecg_normal',
    'restecg_st-t abnormality'
    
]

# Convert the boolean columns to numerical (0 and 1)
data[boolean_columns] = data[boolean_columns].astype(int)

```

Certain features like 'sex' is coded as 'object' in the original dataset. It needs to be to be converted to numeric. 
```ruby


# Convert 'sex' column to numeric: Female -> 0, Male -> 1
data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})

```

Some variables such as slope, exang, and thal are incldued in the dataset but the data dictionary on Kaggle doe not provide the definition of what those variables are. So I removed them for now from the future analysis.

```ruby
data = data.drop(columns=['slope', 'exang', 'thal', 'oldpeak', 'ca'])
```


## Modeling 

Now that we have done  the data cleaning, let's perform the XGBoost. In total afrer the data preparation, we have 17 predictors , excluding the ID column, and one outcome ('num') and 920 patients. To prevent overfitting as we have relatively small dataset, I’ll use parameters like max_depth, eta (learning rate), subsample, and colsample_bytree to control model complexity and reduce overfitting.

1. Let's begin with importing the necessary libraries and prepare the data for a training and  testing set
   
```ruby
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

X = data.drop(columns=['id', 'num'])  # Predictors (all columns except 'id' and 'num')
y = data['num']  # Outcome variable

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. I will then set up XGBoost regressor with hyperparameters to prevent overfitting. Now to know which parameter values work best, we will rely on our good old friend,cross validation, which I have talked about multiple times in my previous post. 
To briefly explain, cross-validation helps in choosing hyperparameters that generalize well to unseen data, reducing the risk of overfitting (model too complex) or underfitting (model too simple).  Boosting algorithms like XGBoost are powerful and can easily overfit if hyperparameters are not carefully tuned. To find the best combination of hyperparameters, we can use techniques like Grid Search in conjunction with cross-validation to test different parameter values.


```ruby
# Set up the initial XGBoost regressor
xgboost_regressor = xgb.XGBRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 4, 5, 6],              # Different depths of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # Different learning rates
    'n_estimators': [50, 100, 150, 200],     # Different numbers of trees

}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgboost_regressor,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric for regression
    cv=5,                              # 5-fold cross-validation
    verbose=1,                         # Output progress
    n_jobs=-1                          # Use all available cores
)
```

To further explain how does the code above work, you can see that The parameter grid defines all combinations of hyperparameters that we want to test. For our  grid:
    max_depth: [3, 4, 5, 6] (4 values)
    learning_rate: [0.01, 0.05, 0.1, 0.2] (4 values)
    n_estimators: [50, 100, 150, 200] (4 values)
   

Based on the values, we have 4×4×4= 64 combinations in total. 

For the cross validation process, cv=5 specifies 5-fold cross-validation. This means that the dataset is split into 5 equal parts (folds). In each iteration, 4 folds are used for training the model, and the remaining 1 fold is used for validation (testing). This process is repeated 5 times so that each fold gets a chance to be the validation set. For each of the 64 combinations of hyperparameters, the model is trained and validated 5 times (once for each fold in the 5-fold cross-validation). For each combination of hyperparameters, GridSearchCV calculates the mean of the performance metric (in this case, negative Mean Squared Error, neg_mean_squared_error) across all 5 folds.

Note that the "neg" prefix indicates that scikit-learn uses negative values because it aims to maximize the score. Since we want to minimize MSE, the model with the highest (least negative) value is the best. 

For example, the irst hyperparameter combination is max_depth=3, learning_rate=0.01, n_estimators=50, subsample=0.8, colsample_bytree=0.8.  The dataset is split into 5 folds. The model is then trained on folds 1-4, and fold 5 is used for validation. The MSE is calculated. The model is trained on folds 1, 2, 3, 5, and fold 4 is used for validation. The MSE is calculated. This process is repeated until each fold is used as the validation set once. The average MSE across these 5 iterations is calculated for this hyperparameter combination. The same thing happens for the second till the lasy hyperparameter combination. In total, we would have 320 iterations (64 combinations × 5 folds).

3. The next step is to run the GridSearchCV on the training data (X_train and y_train) to find the best combination of hyperparameters from the specified param_grid. Then I fit the model using GridSearchCV and get the best parameters and model.
```ruby
# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
```
4. Then, I get the best parameters and model
```ruby
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
```
I found that the best parameters are as follows: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 50}

5. Now that we get the best parameters estimates for our training data, let's evaluate the model ebuilt using the best parameters found by GridSearchCV.
Here, I evaluate the model based on the training data first to see if is potentially overfitting. Then, I will evaluate the model on the test set to evaluate the model's performance on unseen data.

```ruby
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
```

6. Let's take a look at the performance matrix.
```ruby
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
```

In the code snippet above, train_mse and test_mse represent the Mean Squared Error (MSE) on the training and test sets, respectively. A significant difference between train_mse and test_mse might indicate overfitting. train_r2 and test_r2 represent the R-squared score for the training and test sets, respectively. Ideally, these scores should be reasonably close, with a high value indicating that the model explains a large portion of the variance in the data.

7. Now let's interpret the findings.  I have got Training MSE: 0.7160, R2: 0.4508 and Testing MSE: 0.7621, R2: 0.4131. The Training MSE (0.7160) and Testing MSE (0.7621) are relatively close to each other, indicating that the model performs similarly on both the training and test sets. Similarly, the Training R² (0.4508) and Testing R² (0.4131) are close, suggesting that the model has not memorized the training data excessively (which would be a sign of overfitting).

The R² values for both training and testing sets are relatively low, which indicates that the model is not capturing a significant amount of the variance in the data. According to the performance matrix. There is a likelikood that the current model is underfitting. This is not surprised consideirng that the model could be too simple to capture the underlying patterns in the data. The close MSE and R² values between training and testing also imply that the model is not overfitting, but rather it is underfitting.

## Model Improvement 

According to  the results, there are a few things that I can do.

1. Feature Engineering and Hyperparameter Tuning. 

We can try expanding the range of the max_depth or n_estimators parameters in the grid search function, which may allow to the functiion find better combinations of parameters.

2. Use More Advanced Models:

## Conclusion

Now that you have learned how GBT works and practiced with a real world exmaple, it's clear that the algorithm offers several advantages:

1. **Reduced Data Preparation Tasks:**  
   Since GBT uses decision trees as base learners, there is less need for extensive data preprocessing tasks, such as scaling or normalization, which are required by algorithms like K-Nearest Neighbors (KNN). GBTs handle different feature scales inherently well.

2. **Controlled Model Complexity and Overfitting:**  
   GBT allows for fine control over model complexity and helps prevent overfitting by adjusting parameters like the learning rate, tree depth, and regularization methods. This makes it suitable for both small and large datasets.

3. **Flexibility Across Different Types of Outcomes:**  
   The algorithm works with various types of target outcomes, including continuous, categorical, and ranking data, making it highly versatile for different types of machine learning problems.

4. **More Accurate and Robust Predictions:**  
   Like bagging methods, GBT produces a final model that is a weighted sum of all weak models (trees). This ensemble approach often results in more accurate predictions than a single decision tree because it reduces bias while maintaining a reasonable variance, thereby providing a more robust model.

Overall, GBTs are powerful, flexible, and widely applicable for many predictive modeling tasks, offering a balance between simplicity, interpretability, and predictive performance.


For the full code snippets, please feel free to visit my GitHub page


