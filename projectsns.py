import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StudentPerformanceFactors.csv', low_memory = False)
print(df.head())
print(df.info())
print(df['Family_Income'].isnull().sum())
print(df['Distance_from_Home'].isnull().sum())
df['Tutoring_Sessions']= df['Tutoring_Sessions'].astype(np.float64).fillna(np.mean(df['Tutoring_Sessions']))
print(df['Tutoring_Sessions'])

# relation between Hours_Studied and Exam_Score:-
sns.set_style('whitegrid')
sns.scatterplot(data = df, x = 'Hours_Studied', y = 'Exam_Score')
plt.title('Relation between Hours Studied and Exam Score')
plt.savefig('Hours_Studied_vs_Exam_Score.png', dpi = 300 , bbox_inches = 'tight')
plt.show()

# relation numerical data :-
corr = df.corr(numeric_only = True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('Correlation_Heatmap.png', dpi = 300 , bbox_inches = 'tight')
plt.show()

# show score distribution

sns.histplot(df['Exam_Score'], kde = True , color = 'gold')
plt.title('Distribution of Exam Scores')
plt.savefig('Exam_Score_Distribution.png', dpi = 300 , bbox_inches = 'tight')
plt.show()

# relation of attentance and score:-

sns.regplot(data = df, x = 'Attendance', y = 'Exam_Score', scatter_kws = {'alpha':0.5}, line_kws = {'color':'red'})
plt.title('Relation between Attendance Rate and Exam Score')
plt.savefig('Attendance_Rate_vs_Exam_Score.png', dpi = 300 , bbox_inches = 'tight')
plt.show()

# check for outliers in exam scores:-
sns.boxplot(data = df, x = 'Exam_Score')
plt.title('Boxplot of Exam Scores')
plt.savefig('Exam_Score_Boxplot.png', dpi = 300 , bbox_inches = 'tight')
plt.show()

# pair plot:-

sns.pairplot(df)
plt.savefig('Pairplot.png', dpi = 300 , bbox_inches = 'tight')
plt.show()


print('#------------------------------------------------end of code------------------------------------------------#')                                                    