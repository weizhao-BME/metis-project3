# Metis Data Science Bootcamp | Project 3

---

## Predicting Approval Status of Mortgage Refinancing

**A Classification Analysis**

Project timeline: Three weeks

Final presentation is posted [here](https://github.com/weizhao-BME/metis-project2/blob/main/presentation/presentation_project2.pdf).

------------

### **Introduction** 

Many believe that mortgage rate will stop dropping. But, the National Association of Realtors expected rates still keep at a low level as 2020. To date, mortgage rate is near record lows, which signifies an economy that continues to struggle. This rate environment is advantageous for those who are seeking a refinancing to offload the financial burdens.  Therefore, it is a good idea to apply for refinancing as soon as possible in order to secure a low rate. But, it is beneficial to understand what financial features lenders focus on to make decision of approvals and how these features play a role in approving or denying applications. 

This project addressed these questions using a machine learning approach. First, the data was collected from Home Mortgage Disclosure Act (HMDA) query website. Second, a feature selection was performed using a 5-fold cross-validated random forest model after initial data cleaning in order to identify importance financial features. Following this step, a random forest model was trained using all the training data along with selected features. Because random forest model has limited ability to interpret feature importance, a logistic regression model was trained and tuned using selected features based on a 5-fold cross-validation. Finally, the random forest and logistic regression models were used to establish a voting-based ensemble model for a better performance. 

***********************

### **Methods**

The figure below shows the workflow of classification modeling. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/workflow.png" alt="Figure 1" width="600"/>

#### **Data cleaning**

A total of 78620 refinancing applications in Massachusetts  were collected from Consumer Financial Protection Bureau. The data contains 99 features, such as the location of applications, loan type, ethnicity/race, genders, loan amount, HOEPA status, and debt-to-income ratio. Irrelevant features, such as location, ethnicity/race, sex and genders, were excluded from the dataset. The applications with "Nan" values in remaining features were excluded as well. The "action taken" column was used as prediction labels. It contains "loan originated", "application approved but not accepted", and "application denied". The first two conditions were combined and considered as "application approved". Finally, the clean data included 62310 applications (51628 approvals vs. 10682 denials) and 28 financial features. 

The Figure below shows a summary of the raw data during data query. 

<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/data_summary.png" alt="Figure 2" width="600"/>

#### **Model-based feature selection**

It is desirable to reduce the number of features to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model. To this end, the entire the dataset was split into a training and a testing datasets. Using the training dataset, a random forest model was trained and cross-validated (5-fold) to maximize ROC-AUC. Permutation importance was calculated to rank the features, because the impurity-based feature importance reported from the random forest model itself could be misleading for high cardinality features (many unique values) )(Ref: [sklearn doc](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)).  If the permutation importance of a feature is larger than  the 75th percentile of all the feature importance values, this feature was selected as an important feature.

#### **Classification modeling**

For all the training dataset, the selected important features were employed to train a random forest model. Its performance based on an independent testing dataset, in terms of ROC-AUC and confusion matrix were reported. 

Because the features selected based on permutation importance do not inform how lenders make decisions of approval using these feature, a logistic regression model was trained utilizing all the training dataset. This model offers a way to compare the coefficient-based relative feature importance. Similar to the random forest model, the ROC-AUC and confusion matrix were reported. 

----------

### **Results and Discussion**











<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/feature_selection.png" alt="Figure 3" width="600"/>



<img src="https://github.com/weizhao-BME/metis-project3/blob/main/figures/confusion_matrices.png" alt="Figure 4" width="600"/>

