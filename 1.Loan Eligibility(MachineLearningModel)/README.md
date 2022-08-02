# Problem Statement:

Let us say a major bank deals in all home loans and as a process the customer first applies for a home loan, after which the company validates the customer's eligibility to decide to lend loan. Since this is a big bank, hundreds of applications need to be reviewed and approved on a daily basis. Let's assume 1 person process 25 application a day then ideally bank would need to have 4 employees and if one employee average salary is 4500 SGD, it would cost 18000 SGD per month and 216K per annum to a bank to have a team to pick eligible customers to lend loan.

We can automate the eligibility checking process with a ML model based on data gathered from the loan applicant.

# Let’s look at the steps from loading the data to deploying the model. 
To automate the loan eligibility process in realtime based on customer detail let's provided the customer an online application form. And these are the details that customer need to share i.e. Gender, Marital Status, Income, Loan Amount. To automate this process, we need identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

# Feature selection:

| Variable | Description |
| :--------| :----------- | 
| Gender | Male/ Female | 
| Married | Applicant married (Y/N) |
| Monthly Income | In Rupees |
| Loan Amount | In Rupees |

# Online Application form looks as:

    <img width="372" alt="image" src="https://user-images.githubusercontent.com/95873178/182424164-f9743de2-4817-4f8d-af2f-51c6a3422705.png">


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
```diff
# importing machine learning model
from sklearn.linear_model import LogisticRegression
```
```diff
from sklearn.feature_extraction.text import TfidfVectorizer
max_features = 4
tfidf = TfidfVectorizer(max_features=max_features)#stop_words='english',)# norm = None)#)
X1 = tfidf.fit_transform(X)
```
```diff
# training the logistic regression model

model = LogisticRegression() 
model.fit(X.values, y.values)
```
```diff
Gender = 1
Married = 0
ApplicantIncome = .0 
LoanAmount = 600000.0
```
```diff
y_predict = model.predict([[Gender, Married, ApplicantIncome, LoanAmount]])
y_predict
```
```diff
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X.values, y.values)))
```
```diff
# saving the model 
import pickle 
##wb-write in binary mode 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()
```
# Deploying the machine learning model using streamlit

  1. Building the Frontend of the application
  2. Loading and Pre-processing the data
  3. Building the Machine Learning model to automate Loan Eligibility
  4. Deploying the application

## 1. Building the Frontend of the application

1.1 Install Required Libraries<br>
1.2 Creating the Frontend of the app using Streamlit

### 1.1 Install Required Libraries##
```diff
# installing pyngrok
!pip install -q pyngrok
```
```diff
# installing streamlit
!pip install -q streamlit
```
### 1.2. Creating the frontend of the app using streamlit
```diff
# creating the script
%%writefile app.py

# importing required libraries
import pickle
import streamlit as st

# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

# this is the main function in which we define our app  
def main():       
    # header of the page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Check your Loan Eligibility</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True) 

    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female","Other"))
    Married = st.selectbox('Marital Status',("Unmarried","Married","Other")) 
    ApplicantIncome = st.number_input("Monthly Income in Rupees") 
    LoanAmount = st.number_input("Loan Amount in Rupees")
    result =""
      
    # when 'Check' is clicked, make the prediction and store it 
    if st.button("Check"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount) 
        st.success('Your loan is {}'.format(result))
 
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount): 

    # 2. Loading and Pre-processing the data 

    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Married":
        Married = 1
    else:
        Married = 0

    # 3. Building the model to automate Loan Eligibility 

    # if (ApplicantIncome >= 50000):
    #     loan_status = 'Approved'
    # elif (LoanAmount < 500000):
    #     loan_status = 'Approved'
    # else:
    #     loan_status = 'Rejected'
    # return loan_status

    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
     
if __name__=='__main__': 
    main()
 ```
 ## 4. Deploying the application
 ```diff
 # running the app
!streamlit run MLapp.py
```

```diff
!ngrok config add-authtoken 22mgHT1EwrXH9EBP0nyaHLJJ7xE_SFP2HiTjbZPZP8g7jgbn
```
```diff
# making the locally-hosted web application to be publicly accessible
from pyngrok import ngrok

public_url = ngrok.connect('8509')
public_url
```
## 5. In addition I have deployed the model on AWS and Huggingface space
### Huggingface
https://huggingface.co/spaces/ArunkumarCH/LoanApprovalModel
