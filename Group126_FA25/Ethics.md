# Ethics
## Overview
Here, we discuss our model and hyper-parameter choices, and analyze the validity of secondary analyzes.

### Statistical Testing
We used statistical testing to determine if three distributions are same or different and used p value to assess how significant the differences are. While 0.00 is is commonly used as threshold for p value, we used 0.01 to make sure the results are significant enougth to conclude. Moreover, testing showed that we had p values smaller than $10^{-10}$, which is a very strong evidence to draw conclusion.

Before conducting statistical testing, we also made sure the data meets the assumptions of such tests. For example, two sample t test assumes that data are distributed normally, so we made sure by using visualization and transformation. For user statistics which are not normal, we used non-parameteric test which does not require the data to be normal.

### Secondary Analysis
We mainly conducted two analyzes in addition to statistical testing, which are dimensionlity reduction and classification. The purpose of these analyzes are not to train a model for prediction or application, but rather to evaluate or show how separable user groups are. Therefore, we did not spent too much time on choosing different variations of dimesionality reduction methods or classifier including their hyper-paramters.  

## Checklist
[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

### A. Data Collection
- [X] **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?  
  > We are currently using the UCSD Google Local Data where all reviews were collected voluntarily. Thus, no new data would be collected from others people for this project so no opt-in is needed. 

- [X] **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?  
  > The data only contains California businesses so we are limited to that degree of bias. There are also some businesses that contain missing metadata fields which could create biased information due to the nature that the businesses that opted out of providing certain fields could have some correlation with positive/negative reviews. 

- [X] **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?  
  > Yes for now we've processed the data such that we only have a few columns to do our analysis on which are: rating, text, time, gmap_id, user_id, and name. Personable identification like the name field is part of public information on Google which implies they've already given consent. However, even so, we will be working towards removing those fields as we progress in our analysis since it inherently doens't provide contribute to our analysis. 

- [X] **A.4 Downstream bias mitigation**: Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)? 
   > we don't have demographic attributes in the processing of our data. Geographic data like latitude/longitude so far isn't being used in our analysis since our hypothesis and research question isn't inherently geographical. 
    
### B. Data Storage
- [X] **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
   > Yes, the data file is downloaded for each of us to perform analyses privately on our local computer. The data remains on our local computer solely for analysis and project purposes. It shall never be on a public repo, or anywhere on the internet where it is pubic.

- [X] **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?  
   > Since the data we're using is primarily a third party dataset that doesn't get populated as we do our project, we do not know the identities behind the name fields like user_id. However to mitigate this problem overall, we would be focussing on limiting personal identifiable information for our analysis overall.

- [X] **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?
   > The data shall only be used for the duration of the COGS108 project. We will reassess the ethics checklist if we plan on using the data for any reason outside the duration of the course's duration. 

### C. Analysis
- [X] **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
   > We don't see as something relevant for our project at this moment. However, we will keep this in mind as we move forward. 

- [X] **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
   > Yes, in our processed data, we filtered out restaurants with less than 100 reviews with the assumptions that if a restaurants have a lot of reviews, it would likely be a good restaurant might have excluded actual bad restaurants.
    
- [X] **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
   > Yes, we shall avoid cherry-picking restaurants that best represent our data or include visualizations that skew perception. So far we've done very simple descriptive statistics but we will keep in mind as we move forward. 

- [X] **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
   > This ties into the Data collection part but yes, personal identifiable information shall not be used in our analysis like names and user ids. 
    
- [X] **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?
   > Yes we very clearly documented what is done in the checkpoints so far. 

### D. Modeling
- [X] **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
   > We haven't made our model but we will update this when we do. 
    
- [X] **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
   > We haven't made our model but we will update this when we do. 

- [X] **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
   > We haven't made our model but we will update this when we do. 
    
- [X] **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
   > We haven't made our model but we will update this when we do. 

- [X] **D.5 Communicate limitations**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?
   > We haven't made our model but we will update this when we do. 

### E. Deployment
- [X] **E.1 Monitoring and evaluation**: Do we have a clear plan to monitor the model and its impacts after it is deployed (e.g., performance monitoring, regular audit of sample predictions, human review of high-stakes decisions, reviewing downstream impacts of errors or low-confidence decisions, testing for concept drift)?
   > We haven't made our model but we will update this when we do. 

- [X] **E.2 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?  
   > Unless there's any insinuation that reviewing in general implies you're likely spreading negativity, there shouldn't be any harm.
- [X] **E.3 Roll back**: Is there a way to turn off or roll back the model in production if necessary?
   > Yes, our process is well documented on the jupyter-notebook so changes to the model can be made as we assess it in the future. 

- [X] **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?
   > We haven't made our model but we will update this when we do. 
