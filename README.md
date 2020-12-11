# E-commerce Repeat buyers churn prediction using machine learning
Created by Frank Jiang
![Project_image](https://i.pinimg.com/originals/a2/9e/f3/a29ef3c31e530c749c9dcf5451cf8f5b.jpg)
 
 ---
 ### Table of Contents
 
   - [Description](#Description)
   - [Data](#Data)
   - [Feature Engineering](#Feature-Engineering)
   - [Model Fitting](#Model-Fitting)
   - [Cross Validation Algorithm](#Cross-Validation-Algorithm)
   - [Results](#Results)
   - [Reference](#Reference)
   - [Resources](#Resources)
 ---
 
 ## Description
 
 ### Business Interest and background
Merchants sometimes run big promotions (e.g., discounts or cash coupons) on particular dates
(e.g., Boxing-day Sales, "Black Friday" or "Double 11 (Nov 11th)", in order to attract a large
number of new buyers. Unfortunately, many of the attracted buyers are one-time deal hunters,
and these promotions may have a little long-lasting impact on sales. To solve this problem, it is
important for merchants to identify who can be converted into repeated buyers. By targeting
these potential loyal customers, merchants can greatly reduce the promotion cost and enhance
investment (ROI) return.

For more details, you are more than welcome to read our [Final Report](Final Report.pdf)

## Data 

### Data source

Alibaba Cloud provides the data. The data set contains anonymized users' shopping logs in the
past six months before and on the "Double 11" day, and the label information indicating whether
they are repeated buyers. Due to privacy issues, data is sampled in a biased way, so the statistical
result on this data set would deviate from the actual of Tmall.com. Nevertheless, it will not affect
the applicability of the solution.

 [Back To The Top](#Table-of-Contents)
 
### Data Processing
 
 The first challenge we faced is the considerable scale of our data. Our training dataset contains
260864 users’ data, including their profile information and log activities. We failed to read in
data to pandas data frame directly. To solve this problem, we used pandas ‘chunksize = 10000’,
which enable us to process the user_log and user_info data for further analysis.

### Exploratory Data Analysis
Through initial EDA, we are able to find some interesting results.

![User_Profile](Supporting%20Materials/User_Profile.PNG)
*_Figure 1 User Profile Description_

![User_Behavior](Supporting%20Materials/User_Behavior.PNG)
*_Figure 2 User Behavior Description_

![Total_Actions](Supporting%20Materials/Total_Actions.PNG)
*_Figure 3 Total Actions_

![Action_By_Month](Supporting%20Materials/Action_By_Month.PNG)
*_Figure 4 Action By Month Distribution_

 ## Feature Engineering
 We generated new features by aggregating and grouping to generate 5 types of feature: Action Based Feature, 
 Day Based Feature, Product Diversity, User-merchant similarity, and Recent activities.
 
 ![Feature_Engieering](Supporting%20Materials/Feature_Engineering.PNG)
 *_Figure 5 Feature Engineering Example_
 
 This ends with a total of 81 features that are added into the training and testing dataset.
 
 [Back To The Top](#E-commerce Repeat buyers churn prediction using machine learning)

 ## Model Fitting
In this project, we decide to take the traditional approach in model selection and focus on solving the imbalance outcome issues. 
 We fit the following list of model:
 1. Random Forest
 2. Logistic Regression
 3. Gradient Boosting Machine
 4. Extreme Gradient Boosting
 
 In order to avoid the imbalance outcome from overfitting the model, we decide to use the following sampler to balance the data:
 1. SMOTE
 2. Random Under Sampler
 3. ADASYN
 
 ## Cross Validation Algorithm 
 The traditional k-fold cross-validation algorithm would not work as it does not support applying different samplers. Therefore, we
decide to implement a stratified k-fold cross-validation algorithm. We built a simple k-fold
cross-validation algorithm that takes in arguments of the model, data input, data outcome(label),
number of folds, scoring metrics, and sampling method. This algorithm supports all scoring
metrics that are available in ‘sklearn.metrics’ package. Overall, this algorithm computes the
cross-validation score while adjusting for the imbalance in the outcome. 
 
For details regarding the simple algorithm, please see our final report.

 [Back To The Top](#E-commerce Repeat buyers churn prediction using machine learning)

## Results
We use Accuracy and AUC score as our criterion for model evaluation. 

![Performance](Supporting%20Materials/Performance.PNG)
*_Figure 6 Performance under 5-fold cross validation_

**XGBoost with SMOTE as the sampler** has the best performance with the highest accuracy and relatively high AUC. 
Based on the best model, we can explore some business insight from the importance plot.

![Importance_Plot](Supporting%20Materials/Importance_Plot.PNG)
For detailed interpretation on importance feature and actionable insight on predicting repeat-customer, please see our presentation slides.

## Reference
[1] Guimei Liu, Tam T. Nguyen, Gang Zhao. Repeat Buyer Prediction for E-Commerce
https://www.kdd.org/kdd2016/papers/files/adf0160-liuA.pdf

[2] Rahul Bhagat, Srevatsan Muralidharan. Buy It Again: Modeling Repeat Purchase
Recommendations[J]. KDD 2018, August 19-23, 2018, London, United Kingdom
https://assets.amazon.science/40/e5/89556a6341eaa3d7dacc074ff24d/buy-it-again-modelingrepeat-purchase-recommendations.pdf

[3] Huibing Zhang, Junchao Dong. Prediction of Repeat Customers on E-Commerce Platform
Based on Blockchain[J]. Wireless Communications and Mobile Computing Volume 2020,
Article ID 8841437, 15 pages
https://www.hindawi.com/journals/wcmc/2020/8841437/

[4] D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent dirichlet allocation. Journal of Machine
Learning Research, 3(4-5):993–1022, 2003.

[5] L. Breiman. Random forests. Mach. Learn., 45(1):5–32, 2001.

[6] T. Chen and T. He. Xgboost: extreme gradient boosting.
Available on https://github.com/dmlc/xgboost.

[7] M. Dash and H. Liu. Feature selection for classification. Intelligent data analysis, 1(1):131–
156, 1997.

 ## Resources
 The links to the original data source can be found here: https://tianchi.aliyun.com/competition/entrance/231576/information
 
 The links to our presentation for consulting purposes can be found here: 
 [Presentation](#final%20present.pptx) 
 
 [Back To The Top](#E-commerce Repeat buyers churn prediction using machine learning)
 
 
