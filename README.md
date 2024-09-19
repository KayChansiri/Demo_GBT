# The Art of Gradient Boosting Machines: A Practical Approach

Welcome to the third article in my series on ensemble techniques for machine learning. In my previous post, I discussed Random Forest, an ensemble technique that combines decision trees in parallel. In this post, I will introduce you to boosting, another powerful ensemble technique that works sequentially.

There are several boosting methods available, but the most popular by far are AdaBoost (adaptive boosting) and gradient boosting. Today, we’ll focus on gradient boosting, also known as gradient boosted trees (GBT).

Whenever I think of gradient boosting, it reminds me of sculpting a layered structure, like carefully crafting a sandcastle at the beach. The foundation you build influences each layer that follows, and every adjustment contributes to the overall shape of the final castle. It’s an artistic and iterative approach to machine learning, where every decision impacts the next. Let’s dive deeper into the art of gradient boosting together. 

## What is Boosting?

**Understand Weak Learner:** 

Before diving into boosting, let me first introduce the concept of a weak learner. A weak learner, often represented by a single decision tree, can be a classifier or regressor whose predictive power is only slightly better than random guessing. The depth of a weal learner is usually very shallow, ranging from one to five. Despite its simplicity and limited predictive performance, a weak learner becomes valuable when combined with other weak learners to create a more powerful model. It's also important to note that a weak learner is considered adequate as long as its predictions are not completely uncorrelated (perpendicular) with the observed values in a vector space. If they are, the weak learner is no longer useful.

**The Concept of Boosting:** 

Boosting is an ensemble technique where each classifier or regressor tries to correct the errors made by the previous one. Unlike bagging (used in Random Forest), which builds trees in parallel, boosting builds trees sequentially, creating more dependency between the trees in the model. 

**How Gradient Boosting Works:** 

In GBT, the process starts by using a simple model (a weak learner) to make predictions. The residuals (errors) from this model—calculated as the difference between the actual values and predicted values—are then used as the target for the next model. The next model is trained specifically to predict these residuals, and the new predictions are combined with the previous model’s predictions to update the overall prediction. This process continues iteratively, with each new model trying to reduce the residuals further until the overall residuals are minimized. Keep in mind that the concept of adjusted residuals is specific to GBT. AdaBoost as another boosting algorithm relies on weight adjustment to handle misclassified samples, which I will cover in my next post.

**Final Prediction in Boosting:**  

The final prediction depends on what type of boosting we are talking about. For classification GBT, the final prediction calculates the sum of the log odds from the initial prediction and the weighted sum of the log odds from all trees. This cumulative sum is then converted to a probability using a sigmoid function to get the final prediction. I know that concept sounds confusing for now, but bare with me. I promise you will understand it better after you finish reading  the post. For regression tasks, GBT calculate the final prediction as the sum of the initial prediction (e.g., the mean of the outcome) and the weighted sum of the residual corrections from all the trees. In this case, the weighted sum simply refers to the sum of residual corrections across the trees multiplied by the learning rate, which controls how much each tree's residual prediction contributes to the final prediction. Learning rates prevent overfitting by scaling down the updates. For AdaBoost, things work a bit differently. I will not focus on it now though.


We can write a one-line equation representing GBT boosting as below: 

￼<img width="302" alt="F(x) = Fo(x) +" src="https://github.com/user-attachments/assets/d68bc611-51cf-4bd3-9ac1-98026ab9f3a0">

*F(x)* is the final boosted model.

*F<sub>0</sub>(x)* is the initial model (e.g., the mean for regression, the log of the odd for classification).

*h<sub>m</sub>(x)* is the weak learner at iteration 

*η* is the learning rate, which is a parameter that can be fine-tuned.

*M* is the total number of iterations or trees.

## Data Preparation Assumptions for Boosting Techniques

1. **Handling of Scale**: Similar to bagging in Random Forest, boosting algorithms do not require standardization or normalization of the input features as the algorithm is based on decision trees, which are are invariant to the scale of the features. 
2. **Dimensionality:** Unlike certain algorithms that may suffer from the curse of dimensionality such as KNN, boosting can handle high-dimensional data relatively well. However, keep in mind that if the number of features is much larger than the number of samples, the model might suffer from overfitting issues, just like many ML algorithms. Dimensionality reduction techniques (like PCA, EFA) or feature selection based on your project or research objectives can help in such cases. Although GBT works well for high-dimensioanl data, the algorithm does not work well for  high-dimensional sparse data, which this  limitation is similar to other tree-based models.
3. **Data Quality:** Boosting is sensitive to noisy data and outliers because it tries to correct errors from previous learners. If the data is noisy, boosting can overfit to the noise. Some preprocessing, like removing outliers or using regularization techniques, can help mitigate this issue.
4. **Handling Missing Values:** Many boosting algorithms like XGBoost, which I will discuss in my next post, can handle missing values internally. However, it is still a good practice to handle missing values before applying the model to improve model performance. Imputation techniques such as expectation maximization or multiple imputation are helpful if you have missing completely at nonrandom. For data suffering from missing at random and the missing percentage is low, a simpler technique such as single imputation, mean imputation for continuous predictors, or mode imputation for categorical predictors is helpful.
5. **Data Size:** Boosting techniques can be computationally expensive, especially with large datasets, because of their sequential nature. Certain boosting libraries, such as XGBoost or LightGBM provide optimizations for faster computations.


<img width="731" alt="Screenshot 2024-09-18 at 8 04 19 PM" src="https://github.com/user-attachments/assets/97bce814-709b-417d-9533-e70e3d148466">


*Figure 1. The Differences and Workflow of Boosting versus Bagging Algorithms* 

## Types of GBT

GBT can be used with both classification and regression problems, unlike certain boosting algorithms such as AdaBoost that focuses on only classification tasks. For GBT regression problem, the goal is to minimize a loss function, which measures the difference between the predicted values and the actual values. Common loss functions for regression include Mean Squared Error (MSE) and Mean Absolute Error (MAE). For classification problems, the goal is to minimize the log loss or cross-entropy loss. Minimizing log loss is equivalent to maximizing the log-likelihood, which you might have heard if you work with logistic regression before. Both minimizing log loss and maximizing log likelihood are mathematically related and represent the same objective. 

Let's spend some time here for a sec to take a closer look at the concept of log loss. Bascially, log loss measures how well a classification model's predicted probabilities match the actual class labels. It is a measure of error or "loss" that quantifies the distance between the true class labels and the predicted probabilities. The log loss formula for binary classification is as below. A lower log loss means better probability estimates from the model.

<img width="493" alt="Screenshot 2024-09-18 at 9 03 37 PM" src="https://github.com/user-attachments/assets/7a7a75cc-2ac4-4610-a273-df7a2482db0f">


Now, for log-likelihood, the concept measures how likely it is that the observed data (actual labels) would occur given the predicted probabilities by the model. The higher the likelihood, the better the model fits the data. If you are not familiar with the concept. In classification tasks, we aim to maximize the log likelihood because a higher log likelihood indicates that the model is producing predictions that align well with the actual data. The equaiton of the log likelihood can be represented as below: 

<img width="500" alt="Screenshot 2024-09-18 at 9 03 42 PM" src="https://github.com/user-attachments/assets/e7df3058-e4db-41ff-a067-ff5558ab611d">


Notice that the log likelihood is just the negative of the log loss (without the averaging and negative sign). That's why I mentioned previously that the concepts of log loss and log likelihood are mathematically similar.

> On a side note, don't get confused between the concepts of log loss and log of the odds. Those are totally diferent things. The log loss measures the distance between the true labels and the predicted probabilities. The log of the odds (AKA logit) are defined as the ratio of the probability of the event occurring to the probability of it not occurring.

Now that you understand the types of GBT and the relevant concepts, let’s start with GBT for regression.

### 1.GBT for Regression

The core concept behind GBT for regression is gradient descent. If you're unfamiliar with gradient descent, think of it this way: A gradient is a vector that points in the direction of the steepest increase in the loss function (the function we are trying to minimize). The magnitude of the gradient tells us how steep the slope is. In optimization, which is fundamental to machine learning algorithms, we aim to move in the opposite direction of the gradient—the steepest descent—because our goal is to minimize the loss. This is why we talk about gradient descent, focusing on the "negative gradient."

To understand how GBT for regression works, let’s start with the most basic concept— **the initial prediction**. Before we build any decision trees, we begin with a simple, standalone prediction, much like the intercept in a linear regression model. This initial prediction, or we call the initial leaf, is simply the average of the observed outcome across all samples in the dataset.

For example, if you're building a model to predict customer satisfaction, and the average satisfaction score across all customers is 3.4, this initial leaf is set to 3.4. This value stands alone and is not yet associated with any tree—it's just the model's first guess, much like an intercept in a regression model.

Now, imagine your data looks like this:

<img width="347" alt="Screenshot 2024-09-18 at 8 26 08 PM" src="https://github.com/user-attachments/assets/0bf8a7ab-8d87-4e47-a77d-1e545c789318">


*Table 1. Customer Satisfaction Data.*


For each data point, if we subtract the predicted satisfaction score (i.e., 3.4) from the actual satisfaction score, you get the residual. These residuals in the Residual column represent the errors of the current model across customers. Now that we get the initial leaf and the residuals for all participants, let's build the first tree.
 
Unlike trees in Random Forest, the trees we build for GBT predict **residuals** NOT the **observed data**, which in this case is the actual customer satisfaction scores. However, similar to a typical decision tree, at each node in this first tree of the GBT, the algorithm evaluates splits using predictors in the dataset, which could be demographic variables or any variables important for your project, to determine which split best reduces the loss function of the residuals. In this context where the outcome is continuous, the loss function is measured by MSE.
  
For example, suppose we're considering a split on 'Income.' This first tree evaluates different split points (e.g., "Income < $50,000" vs. "Income ≥ $50,000"). For each possible split, it calculates the MSE of the residuals for the data points falling into the left and right child nodes after the split. The split value that results in the lowest total MSE for the residuals is chosen.
  
Still confused? Suppose the algorithm is considering a split on the Income variable, with a threshold of $50,000. This means we create two groups of samples: 1) Group 1 representing customers with Income < $50,000, and 2) Group 2 representing customers with Income ≥ $50,000. For each group, the algorithm computes the mean of the residuals. Again, I emphasize the mean of the residuals **NOT** the actual customer satisfaction scores themselves. For examples, the residuals for Group 1 (Income < $50,000) are [0.6,−0.4,−0.4], the mean residual for this group would be - 0.07. Similarly, the algorithm computes the mean residual for Group 2 in the same waay. 


![Corrected_Income_Residual_Decision_Tree](https://github.com/user-attachments/assets/ff841cde-bccc-4548-ac45-f7a70e962e75)

*Figure 2. The First Weak Learner in the GBT of the Customer Satisfaction Data.*

For each group, The algorithm calculates the Mean Squared Error (MSE) between the actual residuals and the predicted residuals, using the equation below as an example for Group 1:

<img width="619" alt="Screen Shot 2024-08-29 at 4 50 31 PM" src="https://github.com/user-attachments/assets/6dd17a75-9367-4e38-b17b-c1b68c3735f4">

The total MSE for this split is a weighted sum of the MSEs for each group: 

<img width="466" alt="Screen Shot 2024-08-29 at 4 51 31 PM" src="https://github.com/user-attachments/assets/b400fb5a-84f3-4d65-bc7f-32497281cf5f">

Here, *n1* and *n2* represent the number of samples in Group 1 and Group 2, respectively, while *n* is the total number of samples. The split that results in the lowest total MSE is chosen as the best split for that node. For the current fictitious data, let's assume that 50,000 is the best split.

Now that we have the averaged residuals from Tree 1, we will use them to update the predicted value (i.e., customer satisfaction score) for each customer. In the real world, you will likely have more than one predictor. However, I have built the first tree using only income as the predictor here for simplicity. To update the predicted value for each customer, you simply pass the data point down the tree according to the decision rules at each node, find the residual for that value, and add it to the initial leaf value to obtain the new predicted value.

For example, for customer #1 in Table 1, their income is less than 50,000, and the averaged residual for them is -0.07, according to the tree in Figure 2. Thus, the new predicted value is:

**New Predicted Value = Initial Leaf + (Averaged Residual from Tree n)** = 3.4 + (-0.07) = 3.33

Now, we apply this process for every customer, and you will get the updated table below:


<img width="777" alt="Screen Shot 2024-09-17 at 12 07 51 PM" src="https://github.com/user-attachments/assets/d8ee5d54-2e96-42d4-b60e-cbc5b4e590ae">

*Table 2. Customer Satisfaction Data with Adjusted Predicted Values.*

You may notice that the updated predicted values for certain customers, such as customer #5 (3.33), are close to the observed value (3). Isn't this excellent?

The answer is no. We are facing an overfitting issue because Tree 1, as the first weak learner, learns too quickly to predict the observed values, leading to a potential lack of generalizability to new, unseen data. To address this issue, we need to apply a learning rate, a hyperparameter that can be fine-tuned and typically ranges from 0 to 1. Lowering the learning rate scales down the influence of each new tree, reducing the overfitting problem. The equation is as follows:

**New Predicted Value = Initial Leaf + (learning rate × Averaged Residual from Tree *n*)**

With a learning rate of 0.1, the updated predicted value for the first customer is 3.4 + 0.1(−0.07). We apply this for every customer and get the updated table below:


<img width="750" alt="Screen Shot 2024-09-17 at 11 42 15 AM" src="https://github.com/user-attachments/assets/6d5571d4-6e40-4373-9b4c-cbff522172a0">

*Table 3. Customer Satisfaction Data with Adjusted Predicted Values and Learning Rate Application.*


Now, you can see that the predicted values for the first weak learner are less close to the observed values, yet still closer to the actual values than the initial leaf predictions.

With a lower learning rate, the model takes smaller steps toward reducing the loss, requiring more trees to reach a similar level of accuracy. This slower, more controlled process allows the model to capture underlying patterns gradually without fitting to noise. I often found that using more trees with a smaller learning rate can result in a model that generalizes better than a few large steps

The concept I explained above is applicable to building all next trees in the boosting process. In other words, for each new tree, we calculate residuals based on the updated predicted values from the previous tree. Then we use these residuals to continue building trees, each time reducing the model's error further as an iteractive process. The final prediction for any customer is the sum of the initial leaf, adjusted by the weighted residuals from each tree:

**Final Predicted Value=Initial Leaf+(Learning Rate×Residuals from Tree 1)+(Learning Rate×Residuals from Tree 2)+…**

### 2. GBT for Classification

To understand GBT for classification, we need to first grasp the concepts of log odds (logit), odds, probability, and log likelihood. These are foundational concepts of logistic regression, which I explained in my previous [post](https://www.linkedin.com/pulse/understanding-logistic-regression-machine-learning-chansiri-ph-d--v9p7e/?trackingId=dD84NzBIRSa3jT28Z2ATWA%3D%3D).

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

*Table 4. Customer Satisfaction Data.*

The log of the odds can be calculated using the equation:

<img width="553" alt="Log Odds (Logit)" src="https://github.com/user-attachments/assets/41e6f753-20e9-4a06-9d93-de1670cf8401">

In our example, the log of the odds is:

<img width="481" alt="Log Odds (Logit) = log" src="https://github.com/user-attachments/assets/4a973420-821e-4ff4-be3a-203f3ee8ca53">

This logit value is our initial leaf. Now, you may wonder how we use the log odds to get the residual values so that we can use them to grow a tree when the values in our table are just 0 and 1. The answer is that we convert the log odds to a probability so that we can subtract the predicted probability from the actual observed values (0 or 1).
If you recall the equations above, we can get the probability as:


<img width="210" alt="1 + e0 4" src="https://github.com/user-attachments/assets/2f3d6351-fd79-4f33-b9df-6e0a56ebfe0f">

Now that we have the probability as our initial leaf, we can calculate the residuals by subtracting the predicted probability from the actual observed values (0 or 1), just like we did in gradient boosting regression.


<img width="676" alt="Screen Shot 2024-09-17 at 12 38 10 PM" src="https://github.com/user-attachments/assets/6acebd97-40ed-41d6-a7fb-cdce6a18bc5c">

*Table 5. Customer Satisfaction Data with Residuals.*

Similar to what we did earlier in gradient boosting for regression, these residuals will then be used to build our first tree using the same logic we used prevoiusly for GBT for regression. For example, if we want to use income to predict customer satisfaction, the algorithm evaluates various potential cut-off points (e.g., "income < 40,000", "income < 60,000", etc.) and selects the one that minimizes the Gini Index or Cross-Entropy, making the leaf nodes as "pure" as possible in terms of residuals. If "income < 50,000" is considered a potential cut-off point, the residuals of customers with an income below this threshold are considered in the split calculation. The goal is to find the split that results in more homogeneity in terms of their residuals. We can calculate this using the formula below: 


<img width="501" alt="Gini = 1 - (probability of each class)2" src="https://github.com/user-attachments/assets/22b5583f-3742-44cc-9369-757ae888c356">

If the split results in nodes where the residuals are closer to zero, it is considered a good split. The process involves iteratively adjusting the cut-off point until the leaf nodes are as pure as possible, meaning they contain residuals that are minimal and represent better predictions for the classification task. Let’s imagine the first tree we obtained is shown below:

![output-6](https://github.com/user-attachments/assets/36830b57-65d2-4c71-b35f-4de15c7d882c)
*Figure 3. The First Weak Learner in the GBT for Classification of the Customer Satisfaction Data.*

Note that for GBT tasks, including both classification, and regression, residuals are referred to as "pseudo-residuals" because they represent the difference between the actual outcomes and the predicted residuals (i.e., averaged residuals for a regression task and probabilities for a classification task), rather than the difference between the actual and predicted values as in linear regression. A more technical way to explain this is that pseudo-residuals are derived from the gradient of the loss function used, such as the negative log-likelihood (log loss) or binary cross-entropy for classification tasks, and MSE for regression tasks.

Now that we have the residuals as the final leaves of the tree and the initial leaf (e.g., −0.4), we can calculate the predicted probability for each customer.
However, since the initial leaf is in the log of the odds (logit) form and the tree is built based on probabilities, we can't simply sum the log odds and residuals. As explained earlier, log odds and probabilities are two different representations.

To address this issue, we need to perform some mathematical transformations. For each final leaf, we will plug its residual into the formula below to get the adjusted residuals in a form that allows us to combine the value with the initial leaf.

<img width="671" alt="Adjusted Residual" src="https://github.com/user-attachments/assets/d49ea336-2261-47ff-a046-3a32b2adc817">

Let's use Final Leaf 1 in the tree below as an example and apply the formula to calculate the adjusted residual. Note that the previous probability for all customers in this leaf is 0.4 (see Table 4). Thus, for the numerator, we have 0.6+(−0.4)+(−0.4) = −0.2. For the denominator, we calculate (0.4×0.6)+(0.4×0.6)+(0.4×0.6)=0.72. Therefore, the adjusted residual for the first leaf is −0.2/0.72=−0.28. If you repeat this calculation for all final leaves, you will get a tree that looks like the one shown below:

![output-7](https://github.com/user-attachments/assets/4272a8a3-47b1-4827-88dd-236286e7f699)
*Figure 4. The First Weak Learner in the GBT for Classification of the Customer Satisfaction Data with Adjusted Residual.*

Now, we have the final output for each leaf in a form that can be combined with the initial leaf, which is in the log-odds form. To do this, we can use the formula below.

<img width="649" alt="New Log Odds - Initial Log Odds + (Learning Rate × Adjusted Residual)" src="https://github.com/user-attachments/assets/d6d3347b-185d-4220-96ee-648fae5cb0d4">

For Customer 1, we calculate −0.4+(0.1×−0.28). Note that the learning rate can be fine-tuned, but I will use 0.1 for now just for demonstration purposes. We repeat this calculation for every customer to obtain the log-odds for each of them. Then, we convert the log-odds to probabilities so that we can subtract these probabilities from the previous probabilities in the table to get the new set of residuals, as shown in the table below:


<img width="864" alt="Screen Shot 2024-09-17 at 1 00 19 PM" src="https://github.com/user-attachments/assets/e14359b9-c935-4450-b365-e7a5c6e4b49b">

*Table 6. Customer Satisfaction Data with New Predicted Values and Residuals.*

You may notice that the new residuals for some customers are smaller than their initial residuals. This indicates that we are on the right track. However, for certain customers, the new residuals are larger than the original ones. This is why we need to keep repeating the process until all residuals become sufficiently small or until we reach a maximum number of trees specified, indicating that further improvement is minimal. By controlling the learning rate and the number of trees, we can ensure that the model does not overfit the data and generalizes well to unseen data. The ultimate goal is to build a robust model that balances bias and variance. In the end, we will have *n* trees, where *n* is a parameter we fine-tune. 

When we get a new, unseen data point, we will run it through every tree and then sum the final log-odds predictions **across all trees**, along with the initial leaf (-0.4), to obtain the final predicted log-odds for that specific data point. We then convert the log-odds into a probability to obtain the final prediction. Typically, we set the threshold at 0.5, meaning that if someone has a final probability greater than 0.5, their customer satisfaction will be predicted as 1. If it is less than 0.5, it will be predicted as 0.

## Parameters to Fine-Tune in GBT

Now that you understand the concepts of how GBT for regression and classification work, let's talk more about which paramaters can be fine-tuned.

1. **Learning Rate (Range between 0 and 1):**  

GBT include a learning rate (also known as shrinkage or step size) that scales the contribution of each tree. A smaller learning rate (closer to 0) requires more trees to model the data but often results in a more robust model that generalizes better to unseen data. A larger learning rate (closer to 1) may speed up the learning process but risks overfitting if the model captures too much noise from the training data. The choice of learning rate is a trade-off between the number of trees needed and the model's generalization ability.

2. **Tree Depth:**  

The individual trees in GBT are usually shallow, ranging from one to a few levels deep. These shallow trees, again known as "weak learners," are not very powerful on their own but become highly effective when combined through the boosting process. Deeper trees provide more complex decision boundaries, leading to faster learning but also increasing the risk of overfitting, which adds more variance to the model. Conversely, shallower trees take smaller steps in learning (learning more slowly), requiring more iterations to capture the patterns in the data. However, they tend to reduce variance, leading to a more stable model. The optimal depth depends on the complexity of the underlying data.

3. **Regularization Techniques:**  

In addition to the learning rate, GBT can be regularized through other techniques, such as subsampling the training data (also known as stochastic gradient boosting) and controlling the complexity of the trees (e.g., limiting tree depth, minimum samples per leaf). These techniques help prevent overfitting to the training data, especially when dealing with large datasets.

## Example 

Now that you have a solid understanding of GBT, let’s explore a real-world example. I’m using a dataset from Kaggle: Heart Disease Data. This dataset contains 14 attributes, including:
    age,
    sex,
    chest pain type,
    resting blood pressure,
    serum cholesterol,
    fasting blood sugar,
    resting electrocardiographic results,
    maximum heart rate achieved,
    exercise-induced angina,
    number of major vessels, and
    Thalassemia

I recommend visiting [the link](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) for more details on the dataset. The target variable is the 'num' column, which contains five levels indicating the severity of heart disease:

    0: No heart disease
    1: Mild heart disease
    2: Moderate heart disease
    3: Severe heart disease
    4: Critical heart disease

The dataset is relatively small, consisting of 920 instances with 16 features.

## XGBoost For GBT

There are several libraries available for GBT, but I’m using XGBoost for the following reasons:

* **Speed and performance**: XGBoost is known for its efficiency and high performance.
*  **Handling missing values**: Although it’s a good idea to deal with missing values beforehand for better performance, XGBoost handles it effectively without requiring additional preprocessing.
* **Parallel processing**: XGBoost supports parallel processing, making it faster than traditional GBT methods.
* **Versatility**: It can handle both classification and regression tasks, which we’ll explore in this demo.

Other GBT libraries like LightGBM and CatBoost are also great choices. LightGBM, developed by Microsoft, is particularly efficient for large datasets, while CatBoost is great when you work with categorical features, as it avoids extensive preprocessing (e.g., one-hot encoding). Another option is the GradientBoostingRegressor or GradientBoostingClassifier from Scikit-learn. However, these functions tend to be slower compared to XGBoost, especially on larger datasets, and they do not support parallelization as efficiently as XGBoost. Additionally, Scikit-learn requires users to handle missing values manually. As my personal preference is to save the computational time even with a small dataset, I chose to use XGBoost for this demo.

### Data Preparation

Before we get started, we need to ensure that the dataset meets all the assumptions required for GBT, although these assumptions are fewer compared to algorithms like linear regression.

**1. Data Distribution**

GBT models do not require specific assumptions about the distribution of the data, such as normality, homoscedasticity, or linearity. They are designed to capture complex, non-linear relationships between features and the target variable, making them highly flexible and robust for a wide range of data types. Therefore, there is no need to prepare the data based on distributional assumptions.

There is also no need to check for multicollinearity, as GBT models are decision tree-based. These models split data based on feature values rather than fitting coefficients to predictors (as in linear regression), making them generally robust to multicollinearity. Since they don’t assume independence among predictors, GBT models automatically select the most important features at each split, minimizing the impact of correlated features. If two features are highly correlated, the model will usually choose one based on its ability to reduce the loss function.

However, it’s important to note that the assumption of independence of observations still applies to GBT models, just as it does for other machine learning models. While GBT can handle highly correlated features, it may struggle with dependent observations (e.g., in time series data). In cases where observations are dependent, additional steps such as feature engineering or adding lag variables are necessary to account for these dependencies. Fortunately, this does not apply to our current dataset, which is cross-sectional.

**2. Handling of Feature Types**

For XGBoost that we are using in this demo, categorical variables need to be encoded (e.g., using one-hot encoding or label encoding) before training. In our dataset, the 'dataset' variable, which indicates the location of data collection (with 3 levels), and 'cp', which indicates chest pain type (with 3 levels), are categorical. Therefore, we will start by applying one-hot encoding to these variables.

```ruby
data = pd.get_dummies(data, columns=['dataset', 'cp']) 
```

**3. Handling of Missing Values** 

While XGBoost can handle missing values internally, as mentioned earlier, it's generally good practice to address missing values during the data preprocessing stage. In our case, the two columns with missing values—trestbps (resting blood pressure) and chol (cholesterol)—are continuous, and the missing percentages are relatively low. Therefore, I will apply mean imputation for these columns. For the categorical variable 'fbs' (fasting blood sugar), I will use mode imputation to handle the missing data.

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

While GBT algorithms are generally robust to outliers, especially when compared to linear models, extreme outliers can still affect performance by inducing overly specific tree splits. It’s important to note that when discussing outliers, the focus is primarily on continuous predictors, not categorical ones. To identify outliers, I will use the Interquartile Range (IQR) method. Outliers are values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, where Q1 is the first quartile and Q3 is the third quartile. After identifying the outliers, I will use boxplots to visualize them.

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

The output indicates zero outliers for 'age', 28 outliers for 'trestbps', 185 for 'chol', 2 for 'thalch', and 0 for 'num'. The variable with the highest number of outliers is chol (cholesterol). The rest are negligible. I decided not to modify the outliers in cholesterol levels, as the data realistically reflects people at risk for heart disease—a significant number of them would have extremely high cholesterol levels (as shown in the box plot below). When handling outliers in your own project, you should consider whether to address them based on previous literature or the objectives of your project. 



![Screenshot 2024-09-17 at 4 22 22 PM](https://github.com/user-attachments/assets/3a76a2d2-b95a-4dbc-aa56-9bdf332d0599)

**5. Sufficient Sample Size**

While GBTs can perform well even with smaller datasets, having a sufficient sample size is crucial for building robust models. Extremely small sample sizes can lead to overfitting, whereas larger datasets generally benefit from the boosting process, improving model accuracy. In our case, the dataset is on the smaller side. To prevent overfitting, I will adjust and fine-tune certain parameters, as you’ll see later.

**6. Feature Scaling** 

Unlike other algorithms, such as KNN, GBT models do not require feature scaling (e.g., standardization or normalization) because they are based on decision trees, which split data based on feature values rather than their magnitudes. However, scaling might still be useful when features have vastly different ranges, as it can affect interpretability or influence the importance of certain features in the model.

**7. Feature Engineering**

Like other algorithms, GBT models can benefit from feature engineering, where new features are created to capture non-linear relationships or interactions between variables. These newly created variables should be informed by previous literature or rigorous concepts. Since I am not a heart disease researcher, I will skip the feature engineering process for now and focus solely on the existing variables.

**8. Class Imbalance**

When working with GBT for classification tasks, it’s critical to check for class imbalance in the outcome variable. If you encounter class imbalance, consider using techniques like SMOTE (Synthetic Minority Over-sampling Technique), undersampling, or adjusting class weights to improve model performance. I’ve explained more about class imbalance and how to address it in a separate post.

Class imbalance for predictors is typically not a concern. Features can have varying distributions, which is generally acceptable as long as they are useful and predictive of the outcome. However, if there are categorical predictors with rare categories (e.g., more instances of White and Black for the race variable than Asian or Native American), this is not necessarily a class imbalance issue but rather a data sparsity issue. You can address data sparsity by combining rare categories or using specialized encoding techniques like target encoding.


**9. Data Type**

XGBoost and most machine learning algorithms require numerical input data. In this dataset, after applying one-hot encoding, several variables are coded as boolean values. I will convert these binary boolean variables into numerical binary format (e.g., 0 and 1) before using them in XGBoost.

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

Some variables, such as slope, exang, and thal, are included in the dataset, but the data dictionary on Kaggle does not provide definitions for these variables. As a result, I’ve decided to remove them from the analysis for now.

```ruby
data = data.drop(columns=['slope', 'exang', 'thal', 'oldpeak', 'ca'])
```


### Modeling (Regression)

Now that we’ve completed data cleaning, let's move on to performing the XGBoost regression. After data preparation, we have 17 predictors (excluding the ID column) and one outcome (num), with 920 patient records. Since our dataset is relatively small, I’ll use parameters like max_depth and eta (learning rate), to control model complexity. We can also use subsample and colsample_bytree to prevent overfitting by asking the model to select only a portion of the samples to train the tree. However, as we do not have many predictors, I don't think we will have an overfitting issue. I will igmore those commands for now and will add them later if needed.

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

2. Next, I will set up the XGBoost regressor with hyperparameters designed to prevent overfitting or underfitting. To determine which parameter values work best, we’ll rely on cross-validation, a method that I’ve discussed multiple times in previous posts. To briefly recap, cross-validation helps in selecting hyperparameters that generalize well to unseen data, reducing the risk of overfitting (where the model is too complex) or underfitting (where the model is too simple). Boosting algorithms like XGBoost are powerful but can easily overfit if hyperparameters aren’t carefully tuned. To find the optimal combination of hyperparameters, we can use techniques like Grid Search combined with cross-validation to test different parameter values.

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

To further explain how the code above works, the parameter grid defines all the combinations of hyperparameters that we want to test. For our grid:

    max_depth: [3, 4, 5, 6] (4 values)
    learning_rate: [0.01, 0.05, 0.1, 0.2] (4 values)
    n_estimators: [50, 100, 150, 200] (4 values)
   

Based on the values, we have 4 × 4 × 4 = 64 combinations in total.

For the cross-validation process, cv=5 specifies 5-fold cross-validation. This means the dataset is split into 5 equal parts (folds). In each iteration, 4 folds are used to train the model, and the remaining fold is used for validation (testing). This process is repeated 5 times, so each fold is used as the validation set once. For each of the 64 combinations of hyperparameters, the model is trained and validated 5 times (once for each fold).

GridSearchCV calculates the mean of the performance metric (in this case, negative Mean Squared Error, neg_mean_squared_error) across all 5 folds for each hyperparameter combination.

It’s important to note that the "neg" prefix indicates that scikit-learn uses negative values because it seeks to maximize the score. Since our goal is to minimize MSE, the model with the highest (least negative) value will be the best.

For example, the first hyperparameter combination might be max_depth=3, learning_rate=0.01, and n_estimators=50. The dataset is split into 5 folds. The model is trained on folds 1-4, with fold 5 used for validation, and the MSE is calculated. The model is then trained on folds 1, 2, 3, and 5, with fold 4 used for validation, and the MSE is calculated again. This process continues until each fold is used as the validation set once. The average MSE across these 5 iterations is then calculated for this hyperparameter combination.

This same process is repeated for all 64 hyperparameter combinations. In total, there will be 320 iterations (64 combinations × 5 folds).

3. The next step is to run GridSearchCV on the training data (X_train and y_train) to find the best combination of hyperparameters from the specified param_grid. Once the GridSearchCV has been executed, I will fit the model using the best hyperparameters and retrieve the optimal parameters and the best model configuration.

```ruby
# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

```
4. Then, I get the best parameters and model
```ruby
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
```
I found that the best parameters are as follows: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 50}

5. Now that we have the best parameter estimates for our training data, let's evaluate the model built using the best parameters found by GridSearchCV. First, I will evaluate the model on the training data to check for any signs of potential overfitting. Then, I will assess the model's performance on the test set to determine how well it generalizes to unseen data.

```ruby
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
```

6. Let's take a look at the performance matrix.
```ruby

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
```

In the code snippet above,  train_mse and test_mse represent the MSE on the training and test sets, respectively. A significant difference between train_mse and test_mse might indicate overfitting. Additionally, train_r2 and test_r2 represent the R-squared scores for the training and test sets, respectively. Ideally, these R-squared scores should be reasonably close, with higher values indicating that the model explains a large portion of the variance in the data.

7. Now let's interpret the findings. I obtained the following results:

* Training MSE: 0.7160
* Training R²: 0.4508
* Testing MSE: 0.7621
* Testing R²: 0.4131

The Training MSE (0.7160) and Testing MSE (0.7621) are relatively close, indicating that the model performs similarly on both the training and test sets. Similarly, the Training R² (0.4508) and Testing R² (0.4131) are also close, suggesting that the model has not memorized the training data excessively, which would be a sign of overfitting.

However, the R² values for both training and testing sets are relatively low, indicating that the model is not capturing a significant portion of the variance in the data. Based on these performance metrics, it’s likely that the current model is underfitting. This isn't surprising, as with not too many predictors, the model might be too simple to capture the underlying patterns in the data. If overfitting were an issue, we could apply techniques such as using subsamples of features and training sets, similar to the concept of bagging. However, this is not the case here.

### Model Improvement 

Based on the findings, if you encounter similar issues in your personal projects, there are a few strategies you can apply:

**1. Feature Engineering and Hyperparameter Tuning**

You can try expanding the range of parameters, such as *max_depth* or *n_estimators*, in the grid search function. This may allow the function to find better combinations of hyperparameters and improve model performance.

**2. Use More Advanced or Alternative Models**

Consider using more advanced models or algorithms that might better capture the relationships in your data. Options include gradient boosting with additional hyperparameters, or alternative algorithms like Random Forests, Support Vector Machines, or Neural Networks, depending on the nature of your dataset. For this demo, I will convert the outcome variable from continuous to binary to see if the model improves. I will recode the target column num into 0 and 1. Rows coded as 0 (no heart disease) will remain 0 for the new variable. Rows coded as 1 (mild heart disease), 2 (moderate heart disease), 3 (severe heart disease), or 4 (critical heart disease) will all be coded as 1. The reasons for this approach are as follows:

* **Nature of the Target Variable**: When treating *num* as a continuous variable, the model attempts to predict specific values (0, 1, 2, 3, 4), implying varying levels of severity in heart disease. However, these ordinal levels do not necessarily have a linear or direct relationship that a regression model can easily capture. The differences between these classes are not continuous or evenly spaced.

* **Classification is More Suitable for Imbalanced Data**: When num is treated as continuous, the model might struggle with subtle distinctions between closely related values (e.g., 1 vs. 2 or 3 vs. 4). In contrast, classification can more easily differentiate between distinct classes (0 vs. 1), especially when the classes have different distributions.

* **Practicality for Prevention and Treatment**: In practice, treating the outcome as a binary variable (having or not having heart disease) is more meaningful for creating effective prevention and treatment plans, rather than trying to predict the chance of heart disease along a spectrum.
  
```ruby
# Recode the 'num' column: 0 stays as 0, 1, 2, 3, 4 are recoded to 1
data['num'] = data['num'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

data['num'].value_counts()
```
### Modelinng (Classification)
After recoding, we have 411 samples classified as not having heart disease and 509 samples classified as having heart disease. The classes are not extremely imbalanced, which is good for classification tasks. Let's proceed with the classification modeling.

1. Like in the regression task, we will start by separating the predictors (features) from the outcome variable and splitting the data into training and testing sets.
```ruby
# Separate predictors (features) and the outcome variable
X = data.drop(columns=['id', 'num'])  # Use all columns except 'id' and 'num' as predictors
y = data['num']  # Outcome variable

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. Next, we will set up the initial XGBoost classifier and define the parameter grid for hyperparameter tuning.

```ruby

#Set up the initial XGBoost classifier
xgboost_classifier = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 4, 5, 6],              # Different depths of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # Different learning rates
    'n_estimators': [50, 100, 150, 200],     # Different numbers of trees
}
```

3. The next step is to set up a grid search function and print the best parameters.
```ruby

#Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgboost_classifier,
    param_grid=param_grid,
    scoring='accuracy',   # Use accuracy as the scoring metric for classification
    cv=5,                 # 5-fold cross-validation
    verbose=1,            # Output progress
    n_jobs=-1             # Use all available cores
)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")

```

The output indicates that the following as the best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
4. Now, let’s evaluate the model. For classification tasks, the evaluation metrics differ from those used in regression. Instead of using MSE, we will rely on accuracy and the confusion matrix (as I’ve discussed in my previous [post](https://github.com/KayChansiri/Demo_Performance_Metrics)) to assess how well the model performs.
```ruby
#Evaluate the best model on the training and test sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate model performance
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Detailed evaluation metrics
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_pred_test))

```

Here is the output: 


<img width="518" alt="Screenshot 2024-09-17 at 8 14 56 PM" src="https://github.com/user-attachments/assets/f99f1c0e-3704-49d5-ba8a-98afe2acf123">


The accuracy scores indicate that the model performs well on both the training and testing data, with similar accuracy levels. This suggests that the model is not overfitting to the training data and generalizes well to unseen data.

In the classification report, for class 1 (heart disease), the precision is 0.87, meaning that of all instances predicted as class 1, 87% are correctly classified. The recall is 0.80, meaning that of all actual instances of class 1, 80% are correctly identified. The F1-score of 0.83 indicates a good balance between precision and recall for class 1.

The confusion matrix provides further insight:

**True Negatives (TN)**: These are the cases where the model correctly predicted "no heart disease." For example, out of all the people who actually don't have heart disease, the model correctly identified 75 of them.

**False Negatives (FN)**: These are the cases where the model incorrectly predicted "no heart disease" when the person actually has it. For example, 34 people who actually have heart disease were mistakenly classified as not having it by the model.

**True Positives (TP)**: These are the cases where the model correctly predicted "heart disease." In this case, 87 people who actually have heart disease were correctly identified by the model.

**False Positives (FP)**: These are the cases where the model incorrectly predicted "heart disease" for people who don’t actually have it. In this instance, 22 people who don’t have heart disease were mistakenly identified as having it.

### Feature Importance 
Now that we know the classification model performs better than the regression one, let’s explore feature importance. This will help us identify which variables have the most influence on the likelihood of someone having heart disease.

```ruby
feature_importances = best_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)

# Plot the Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in XGBoost Model')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()

```
Here is the output: 

<img width="1041" alt="Screenshot 2024-09-17 at 8 21 13 PM" src="https://github.com/user-attachments/assets/d984c7c3-3580-41a9-9650-9b088e036325">

The output shows that cp_asymptomatic (asymptomatic chest pain) is the most significant predictor of heart disease, meaning it greatly influences the model's predictions. Other important features include dataset_VA Long Beach (indicating patients from the Long Beach area), cp_atypical angina (another chest pain type), sex, chol (cholesterol level), and thalch (maximum heart rate achieved), which have a moderate impact on the model’s decisions.

Less influential features include age and the dataset origins like dataset_Switzerland and dataset_Hungary. Some features, such as dataset_Cleveland, fbs (fasting blood sugar), and certain ECG results, have zero importance, suggesting they do not add any predictive value to the model.



