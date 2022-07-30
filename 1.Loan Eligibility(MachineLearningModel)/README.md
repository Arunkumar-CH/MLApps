### Problem Statement:

Let us say a major bank deals in all home loans and as a process the customer first applies for a home loan, after which the company validates the customer's eligibility to decide to lend loan. Since this is a big bank, hundreds of applications need to be reviewed and approved on a daily basis. Let's assume 1 person process 25 application a day then ideally bank would need to have 4 employees and if one employee average salary is 4500 SGD, it would cost 18000 SGD per month and 216K per annum to a bank to have a team to pick eligible customers to lend loan.

We can automate the eligibility checking process with a ML model based on data gathered from the loan applicant.

### Let's 
To automate the loan eligibility process in realtime based on customer detail let's provided the customer an online application form. And these are the details that customer need to share i.e. Gender, Marital Status, Income, Loan Amount. To automate this process, we need identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

### Feature selection:
# Variable	Description
Gender	Male/ Female
Married	Applicant married (Y/N)
Dependents	Number of dependents
Education	Applicant Education (Graduate/ Under Graduate)
Self_Employed	Self employed (Y/N)
ApplicantIncome	Applicant income
CoapplicantIncome	Coapplicant income
LoanAmount	Loan amount in thousands
Loan_Amount_Term	Term of loan in months
Credit_History	credit history meets guidelines
Property_Area	Urban/ Semi Urban/ Rural
Loan_Status	Loan approved (Y/N)

Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
