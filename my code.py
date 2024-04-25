
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


file_path = 'C:/Users/Adib/Desktop/courses in Trier/NLP/train.tsv'
df = pd.read_csv(file_path, sep='\t')


train_data, test_data, train_labels, test_labels = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)



vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)


model = LogisticRegression()
model.fit(X_train, train_labels)


predictions = model.predict(X_test)


macro_f1 = f1_score(test_labels, predictions, average='macro')
print(f'Macro F1 Score on Validation Set: {macro_f1}')


