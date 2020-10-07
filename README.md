I used UCI’s YouTube Spam Collection Data Set for this project. I started off with exploring various meta-features of the comments. I performed hypothesis testing to see if there is any relation between no. of words, average length of words and text-standard of comment with the comment being spam or not. I analyzed it’s text-features by converting the comments into a term-document matrix. I also looked into most predictive words for spam and ham comments. I noticed words related to “money making” were on the list of top 15 most predictive words for spam comments. Upon diving deeper, I found that those comments wanted people to think that they can make lots of money working from the comfort of their homes. 
After exploring the data, I moved to the machine learning part. I grid searched Countvectorizer and Tf-IDf  vectorizer to convert the comments into document term matrix. Then, I grid searched 4 models ( Logistic regression, Multinomial Naive Bayes,  random forest, XGBoost) for hyper parameter tuning. Among the 4 classifiers, Logistic regression performed the best.
In any machine learning model, the default threshold is set to 0.5 meaning any value above that gets classified as Class A and any value below gets classified as Class B'. However, this can be adjusted depending on the business requirement. In this case, I wanted to give more weight to precision than recall hence I used Fbeta score as a scoring metric. With Fbeta score, I could change the value of beta to 0.5 to give more weight to precision than 


## [Reference](#table-of-contents)
1. Data Source 
   - [UCI’s YouTube Spam Collection Data Set](http://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection#)  
2. [Data Wrangling](https://github.com/Preeti24/Youtube-comments/blob/master/Feature%20Engineering.ipynb)
3. [Exploratory Data Analysis](https://github.com/Preeti24/Youtube-comments/blob/master/Exploratory%20Data%20Analysis.ipynb)
   - [Most predictive words](https://github.com/Preeti24/Youtube-comments/blob/master/Most%20Predictive%20Words.ipynb)
3. [Hypothesis Testing](https://github.com/Preeti24/Youtube-comments/blob/master/Hypothesis%20Testing.ipynb)
4. [Machine Learning](https://github.com/Preeti24/Youtube-comments/blob/master/Machine%20Learning.ipynb)
5. [Helper file](https://github.com/Preeti24/Youtube-comments/blob/master/Helper.py)
6. [Project Report](https://github.com/Preeti24/Youtube-comments/blob/master/Reports/Capstone%202-%20Final%20Report.pdf)
