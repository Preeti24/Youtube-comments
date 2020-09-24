import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(15,5)

class analysisHelper:
    def __init__(self,gs,X_test,y_test,beta,isChart=False):
            self.gs=gs
            self.X_test=X_test
            self.y_test=y_test
            self.beta=beta
            self.isChart=isChart
            
    def predictNewThreshold(self,predict_proba,threshold):
        predict_newThreshold=[]
        for prob in predict_proba:
            if prob<threshold:
                predict_newThreshold.append(0)
            else:
                predict_newThreshold.append(1)
        return predict_newThreshold

    def calculateMetrics(self,cm,beta=0.5):
        tp = cm[0][0] 
        fp = cm[0][1] 
        fn = cm[1][0] 
        tn = cm[1][1]

        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=2*(precision*recall)/(precision+recall)
        FBeta=((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
        accuracy=(tp + tn) /(tp + tn + fp + fn)

        return precision,recall,F1,FBeta,accuracy

    def calculateMetricsForAdjustedThreshold(self,gs,X_test,y_test,beta=0.5):
        predict_proba=gs.predict_proba(X_test['CleanWordList'])[:,1]
        threshold=np.arange(0,1,.01)

        Precision=[]
        F1=[]
        FBeta=[]
        Accuracy=[]

        for t in threshold:
            cm=confusion_matrix(y_test,self.predictNewThreshold(predict_proba,t))
            Precision.append(self.calculateMetrics(cm)[0])
            F1.append(self.calculateMetrics(cm,beta)[2])
            FBeta.append(self.calculateMetrics(cm,beta)[3])
            Accuracy.append(self.calculateMetrics(cm)[4])    
        metric_df=pd.DataFrame({'Threshold':threshold,'Precision':Precision,'F1':F1,'FBeta':FBeta,'Accuracy':Accuracy})
        metric_df.reset_index(drop=True,inplace=True)
        return metric_df

    def calculateMetricsForAdjustedBeta(self,gs,X_test,y_test,beta=0.5):
        df=pd.DataFrame(columns=['Threshold','Precision','F1','FBeta','Accuracy','Beta'])
        metric_df=self.calculateMetricsForAdjustedThreshold(gs,X_test,y_test,beta)
        metric_df['Beta']=beta
        df=pd.concat([df,metric_df])
        df.fillna(0,inplace=True)
        return df

    def plotThresholdvsMetric_adjustedBeta(self,gs,X_test,y_test,beta=0.5,isChart=False):
        metric_df=self.calculateMetricsForAdjustedBeta(gs,X_test,y_test,beta)
        xmark=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['Threshold'].iloc[0]
        ymark=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['FBeta'].iloc[0]
        beta=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['Beta'].iloc[0]

        if isChart==True:
                text= "Threshold={:.2f},\nFBeta={:.4f}".format(xmark,ymark)

                sns.lineplot(x='Threshold',y='FBeta',data=metric_df,)
                plt.axvline(x=xmark,ymin=0,ymax=1,color='red')
                plt.annotate(s=text,xy=(xmark,ymark))
                plt.ylabel('FBeta Score')
                title='Threshold vs FBeta Score'
                plt.title(title);
        return beta,xmark,ymark
