# Ensemble Learning - SEL 

For the second project in the Supervised and Experiential Learning course, we conducted an implementation and validation of two ensemble learning algorithms, the Random Forest and the Decision Forest.

The structure of the work is 
```bash
PW1 SEL - RISE
├── data
│   ├── breast.csv
│   ├── heart.csv
│   ├── obesity.csv 
├── source
│   ├── ensemble_learning.py
│   ├── experiments.py
│   └── main.py
├── results
│   ├── heart
│   │   ├── rf
│   │   │   ├── confusion_matrix.svg
│   │   │   ├── classification_report.txt
│   │   │   └── train.txt
│   │   └── df
│   │          ├── confusion_matrix.svg
│   │          ├── classification_report.txt
│   │          └── train.txt
│   ├── breast
│   │   ├── rf
│   │   │   ├── confusion_matrix.svg
│   │   │   ├── classification_report.txt
│   │   │   └── train.txt
│   │   └── df
│   │         ├── confusion_matrix.svg
│   │         ├── classification_report.txt
│   │         └── train.txt
│   └── obesity
│          ├── rf
│          │   ├── confusion_matrix.svg
│          │   ├── classification_report.txt
│          │   └── train.txt
│          └── df
│                 ├── confusion_matrix.svg
│                 ├── classification_report.txt
│                 └── train.txt
├── documentation
│   └── report.pdf
└── readme.md
```

Use the `main.py` file or the following code: 
```python
from ensemble_learning import * 

cat_variables_heart = ['sex', 'chest', 'sugar', 'ecg', 'angina', 'slope', 'thal', 'disease']

model = RandomForestClassifier(dataset_name='heart', target_name='disease', cat_variables=cat_variables_heart,
             test_split=0.2, n_estimators=75, number_features=2)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)
```
