# Metis Data Science Bootcamp | Project 3

---

## Predicting Approval Status of Mortgage Refinancing

**A Classification Analysis**

Project timeline: Three weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf).

------------

### **Introduction** 

Many believe that mortgage rate will stop dropping. The National Association of Realtors expected rates still keep at a low level as 2020. To date, mortgage rate is near record lows, which signifies an economy that continues to struggle. This rate environment is advantageous for those who are seeking a refinancing to offload the financial burdens. Therefore, it is a good idea to apply for refinancing as soon as possible in order to secure a low rate. It is beneficial to understand what financial features lenders focus on to make decisions of approval and how these features play a role in approving or denying applications. 

This project addressed these questions using a machine learning approach. First, the data was collected from Home Mortgage Disclosure Act (HMDA) query website. Second, a feature selection was performed using a 5-fold cross-validated random forest model after initial data cleaning in order to identify important financial features. Following this step, a random forest model was trained using all the training data along with selected features. Because random forest model has limited ability to interpret feature importance, a logistic regression model was trained and tuned using selected features based on a 5-fold cross-validation. Finally, the random forest and logistic regression models were used to establish a voting-based ensemble model for a balanced performance. 

***********************

### **Methods**

The figure below shows the workflow of classification modeling. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/workflow.png" alt="Figure 1" width="600"/>

#### **Data cleaning**

A total of 78620 refinancing applications in Massachusetts  were collected from Consumer Financial Protection Bureau. The data contains 99 features, such as the location of applications, loan type, ethnicity/race, genders, loan amount, HOEPA status, and debt-to-income ratio. Irrelevant features, such as location, ethnicity/race, sex and genders, were excluded from the dataset. The applications with "Nan" values in remaining features were excluded as well. The "action taken" column was used as prediction labels. It contains "loan originated", "application approved but not accepted", and "application denied". The first two conditions were combined and considered as "application approved". Finally, the clean data included 62310 applications (51628 approvals vs. 10682 denials) and 28 financial features. 

The Figure below shows a summary of the raw data during data query. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/data_summary.png" alt="Figure 2" width="600"/>

#### **Model-based feature selection**

It is desirable to reduce the number of features to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model. To this end, the entire dataset was split into training and testing datasets. Using the training dataset (further split into training and validation datasets), a random forest model was trained and cross-validated (5-fold) to maximize ROC-AUC based on validation validation dataset. During the cross-validation, a random grid search approach was adopted to tune the hyper parameters of the model. With similar performance to that of grid search, random grid search has less computational cost. The hyperparameters and their ranges were adopted as below:

The "n_estimators" ranged from 100 to 500 with an increment of 100. The "max_features" included "auto", "sqrt", and "log2". The "max_depth" ranged from 10 to 50 with an increment of 10. The "min_samples_split" ranged from 2 to 20 with an increment of 5. The "min_samples_leaf" includes 1, 2, and 4. The "bootstrap" includes "True" and "False". 

Permutation importance was calculated to rank the features, because the impurity-based feature importance reported from the random forest model itself could be misleading for high cardinality features (many unique values) )(Ref: [sklearn doc](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)).  If the permutation importance of a feature is larger than the 75th percentile of all the feature importance values, this feature is selected as an important feature.

#### **Classification modeling**

For all the training data, the selected important features were employed to train a random forest model. Its performance based on an independent testing dataset, in terms of ROC-AUC and confusion matrix were reported. 

Because the features selected based on permutation importance do not inform how lenders make decisions of approval using these features, a logistic regression model was trained and cross-validated (5-fold) utilizing the existing training and validation datasets. The regularization parameter, penalty strength("C" (list of [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0])),  was tuned to maximize the model performance. Finally, using the best "C", the same selected features and all the training data, a logistic regression model was trained and further tested using the independent testing dataset. This offers a way of investigating the coefficient-based relative feature importance. Similar to the random forest model, the ROC-AUC and confusion matrix were reported. 

----------

### **Results and Discussion**

The top seven most important features were identified using the threshold of 75th percentile permutation importance value. The selected features were HOEPA status, occupancy type, debt-to-income ratio ,open-end line of credit, business or commercial purpose, derived load product type and lien status (Figure below). It suggests that lenders typically focus on these financial features to make decisions of approval. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/feature_selection.png" alt="Figure 3" width="600"/>

Coefficient-based relative feature importance reported from logistic regression model is shown below. It indicates that lenders are more likely to approve applications if:

- The loan is not a high-cost mortgage or is subject to HOEPA regulations
- The property is used as second residence or investment property
- The property is primarily used for a business or commercial purpose
- The loan is not an open-end (or close-end) line of credit

and are more likely to reject applications if:

- The load is not subject to HOEPA regulations 
- The property is used as principal residence
- The applicants have high debt-to-income ratios
- The property is not primarily used for a business or commercial purpose
- The loan is an open-end line of credit 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/rel_feat_importance_logreg.svg" alt="Figure 4" width="600"/>

Both the random forest and logistic regression models achieved an ROC-AUC of 0.99. The former performed better in predicting denials, while the latter performed better in predicting approvals. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/confusion_matrices.png" alt="Figure 5" width="800"/>

Finally, the tuned random forest and logistic regression models were used to establish a ensemble model using a soft averaging approach. It achieved an ROC-AUC of 0.99 and a relatively balanced performance between the random forest and logistic regression models.  The figure below shows the confusion matrix of the ensemble model.

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/cf_mat_voting.svg" alt="Figure 6" width="450"/>

------

#### **Conclusion**

In this analysis, important features that lenders focus on to make decisions of approval were identified. And how lenders use these features as a potential guideline to make decisions of approval were explored. In the future, this work may be extended to employ nationwide data with more loan purposes. A more sophisticated deep neural network could be leveraged for a more complex classification. 

