## Youtube comments classification
In this project I tried to analyze comments from music videos on YouTube. I started off with exploring the data and ran a few hypothesis tests to see if my intuition about the data was right. I built a machine learning model which  predicts, by analysing text and meta features, if the comment is Spam or Ham. 
The comments section can be a gold-mine of information if dug deeper but this functionality is not available in YouTube studio yet. In this project I tried to bridge this gap by providing insight into the content of the comments. Along with classifying the comments into 2 categories, I also explored what words were most predictive for each category. This helped me to know why a certain comment was classified as Spam or not.  It was important for me to not miss any good comment by flagging it as Spam so I adjusted my model accordingly to consider this requirement. At the end, I analyzed the wrong classifications from the model and found that most of the misclassified Spam comments were written in a way that didn't use any “spammy words” and hence slipped through the spam filter. On the contrary, misclassified Ham comments seemed very similar to spam comments to me. The dataset used for this project is manually labelled and I guess few comments  got mislabelled by mistake. 


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
