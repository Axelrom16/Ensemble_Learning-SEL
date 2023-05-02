from ensemble_learning import *


cat_variables_heart = ['sex', 'chest', 'sugar', 'ecg', 'angina', 'slope', 'vessels', 'thal', 'disease']
cat_variables_breast = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                        'Mitoses', 'Class']
cat_variables_obesity = ['Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
                             'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']


####
# Random Forest
####
# Heart data set
model = RandomForestClassifier(dataset_name='heart', target_name='disease', cat_variables=cat_variables_heart,
             test_split=0.2, n_estimators=75, number_features=2)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)

# Breast data set
model = RandomForestClassifier(dataset_name='breast', target_name='Class', cat_variables=cat_variables_breast,
             test_split=0.2, n_estimators=75, number_features=4)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)

# Obesity data set
model = RandomForestClassifier(dataset_name='obesity', target_name='NObeyesdad', cat_variables=cat_variables_obesity,
             test_split=0.2, n_estimators=10, number_features=4)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)


####
# Decision Forest
####
# Heart data set
model = DecisionForestClassifier(dataset_name='heart', target_name='disease', cat_variables=cat_variables_heart,
             test_split=0.2, n_estimators=25, number_features=6)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)

# Breast data set
model = DecisionForestClassifier(dataset_name='breast', target_name='Class', cat_variables=cat_variables_breast,
             test_split=0.2, n_estimators=25, number_features='runif')
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)

# Obesity data set
model = DecisionForestClassifier(dataset_name='obesity', target_name='NObeyesdad', cat_variables=cat_variables_obesity,
             test_split=0.2, n_estimators=50, number_features=4)
X_train, y_train, X_val, y_val = model.import_data()
model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = model.evaluate(X_val, y_val)
print(pred)
print(acc)


