# Problem Statement:

Let us say a major bank deals in all home loans and as a process the customer first applies for a home loan, after which the company validates the customer's eligibility to decide to lend loan. Since this is a big bank, hundreds of applications need to be reviewed and approved on a daily basis. Let's assume 1 person process 25 application a day then ideally bank would need to have 4 employees and if one employee average salary is 4500 SGD, it would cost 18000 SGD per month and 216K per annum to a bank to have a team to pick eligible customers to lend loan.

We can automate the eligibility checking process with a ML model based on data gathered from the loan applicant.

# Let's 
To automate the loan eligibility process in realtime based on customer detail let's provided the customer an online application form. And these are the details that customer need to share i.e. Gender, Marital Status, Income, Loan Amount. To automate this process, we need identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

# Feature selection:

| Variable | Description |
| :--------| :----------- | 
| Gender | Male/ Female | 
| Married | Applicant married (Y/N) |
| Monthly Income | In Rupees |
| Loan Amount | In Rupees |


# Loan Eligibility Prediction
## Steps involved:
  1. Loading the dataset
  2. Pre-processing the dataset
  3. Building the Loan Prediction model

## 1. Loading the dataset
```diff
#importing libraries

import pandas as pd

#loading the dataset
train = pd.read_csv('C:\\Users\\Arunkumar\\OneDrive\\Documents\\Data Science\\Deploying Machine Learning model using Streamlit\\loan_data.csv')
train.head()
```
## 2. Pre-processing the dataset
```diff
#converting categories into numbers
train['Gender']= train['Gender'].map({'Male':0, 'Female':1})
train['Married']= train['Married'].map({'No':0, 'Yes':1})
train['Loan_Status']= train['Loan_Status'].map({'N':0, 'Y':1})
```
```
train.head()
X.head()
y.head()
```
## 3. Building the Loan Prediction model
