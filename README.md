## Youtube comments classification

## App Demo  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/preetiprajapati/youtubecommentanalyzer/main/app.py)

![](Demo_Gif.gif)

In this project I tried to analyze comments from music videos on YouTube. I started off with exploring the data and ran a few hypothesis tests to see if my intuition about the data was right. I built a machine learning model which  predicts, by analysing text and meta features, if the comment is Spam or Ham. 

The comments section can be a gold-mine of information if dug deeper but this functionality is not available in YouTube studio yet. In this project I tried to bridge this gap by providing insight into the content of the comments. Along with classifying the comments into 2 categories, I also explored what words were most predictive for each category. This helped me to know why a certain comment was classified as Spam or not.  It was important for me to not miss any good comment by flagging it as Spam so I adjusted my model accordingly to consider this requirement. At the end, I analyzed the wrong classifications from the model and found that most of the misclassified Spam comments were written in a way that didn't use any “spammy words” and hence slipped through the spam filter. On the contrary, misclassified Ham comments seemed very similar to spam comments to me. The dataset used for this project is manually labelled and I guess few comments  got mislabelled by mistake. 

To be able to access the model in real-time, I create an app using streamlit. This app scrapes YouTube comments using YouTube API and classifies them into positive/negative and spam/non-spam comments. The noun phrases are helpful in understanding the context and help infer what is being talked about in the sentence without having to read the full text. 
I overrode Spacy's methods to highlight chunks of Noun Phrases in the comments and display sample comments containing those phrases to help users get context of the comments. 



## [Reference](#table-of-contents)
1. Reports
   - [Project proposal](https://github.com/Preeti24/Youtube-comments/blob/master/Reports/Capstone%202%20-Project%20Proposal.docx)
   - [Slide Deck](https://github.com/Preeti24/Youtube-comments/blob/master/Reports/SlideDeck.pdf)
   - [Final report](https://github.com/Preeti24/Youtube-comments/blob/master/Reports/Capstone%202-%20Final%20Report.pdf)
1. Project Files
   - [Data Source - UCI’s YouTube Spam Collection Data Set](http://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection#)  
   - [Data Wrangling](https://github.com/Preeti24/Youtube-comments/blob/master/Feature%20Engineering.ipynb)
   - [Exploratory Data Analysis](https://github.com/Preeti24/Youtube-comments/blob/master/Exploratory%20Data%20Analysis.ipynb)
      - [Most predictive words](https://github.com/Preeti24/Youtube-comments/blob/master/Most%20Predictive%20Words.ipynb)
   - [Hypothesis Testing](https://github.com/Preeti24/Youtube-comments/blob/master/Hypothesis%20Testing.ipynb)
   - [Machine Learning](https://github.com/Preeti24/Youtube-comments/blob/master/Machine%20Learning.ipynb)
   - [Helper file](https://github.com/Preeti24/Youtube-comments/blob/master/Helper.py)
  
