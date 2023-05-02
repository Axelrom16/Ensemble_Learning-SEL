import matplotlib.pyplot as plt
from ensemble_learning import *
import math
import time
from datetime import timedelta


# - Utils -
def calculate_nf(x, random_forest=True):

    if random_forest:
        # Random forest
        nf = [1, 2, round(math.log2(x) + 1), round(math.sqrt(x))]
        return np.unique(nf)
    else:
        # Decision tree
        nf = [round(x/4), round(x/2), round(3*(x/4)), 'runif']
        return np.unique(nf)


# - Data sets -
iris_dataset = {
    'name': 'iris',
    'target_variable': 'variety',
    'cat_variables': ['variety'],
    'n_features': 4,
    'nt': [1, 10, 25, 50, 75, 100],
    'nf_rf': calculate_nf(4),
    'nf_df': calculate_nf(4, random_forest=False)
}
heart_dataset = {
    'name': 'heart',
    'target_variable': 'disease',
    'cat_variables': ['sex', 'chest', 'sugar', 'ecg', 'angina', 'slope', 'vessels', 'thal', 'disease'],
    'n_features': 13,
    'nt': [1, 10, 25, 50, 75, 100],
    'nf_rf': calculate_nf(13),
    'nf_df': calculate_nf(13, random_forest=False)
}
breast_dataset = {
    'name': 'breast',
    'target_variable': 'Class',
    'cat_variables': ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                        'Mitoses', 'Class'],
    'n_features': 9,
    'nt': [1, 10, 25, 50, 75, 100],
    'nf_rf': calculate_nf(9),
    'nf_df': calculate_nf(9, random_forest=False)
}
obesity_dataset = {
    'name': 'obesity',
    'target_variable': 'NObeyesdad',
    'cat_variables': ['Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
                             'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'],
    'n_features': 16,
    'nt': [1, 10, 25, 50, 75, 100],
    'nf_rf': calculate_nf(16),
    'nf_df': calculate_nf(16, random_forest=False)
}

datasets = [iris_dataset, heart_dataset, breast_dataset, obesity_dataset]


########
# - Random Forest -
########
for dataset in datasets:

    print("\n###########")
    print("Data set:", dataset['name'])
    print("###########")

    tree_file_rules = open('../results/' + dataset['name'] + '/rf/' + dataset['name'] + '_trees_rf_rules.txt', "w")
    tree_file = open('../results/' + dataset['name'] + '/rf/' + dataset['name'] + '_trees_rf.txt', "w")
    nt_vec = []
    f_vec = []
    acc_vec = []
    train_acc_vec = []
    time_vec = []
    for nt in dataset['nt']:
        for f in dataset['nf_rf']:
            start_time = time.monotonic()
            model = RandomForestClassifier(dataset_name=dataset['name'], target_name=dataset['target_variable'],
                                           cat_variables=dataset['cat_variables'],
                                           test_split=0.2, n_estimators=nt, number_features=f)
            X_train, y_train, X_val, y_val = model.import_data()
            model.fit(X_train, y_train)
            final_trees = model.show_trees()
            pred = model.predict(X_val)
            acc_train, acc = model.evaluate(X_val, y_val)
            feat_importance = model.feature_importance()
            end_time = time.monotonic()

            print("Number of trees:", str(nt))
            print("Number of features:", str(f))
            print("Train accuracy:", acc_train)
            print("Test accuracy:", acc)
            print("Feature importance:", feat_importance)

            tree_file.write("\n#-------\n")
            tree_file_rules.write("\n#-------\n")
            tree_file.write("Number of trees: " + str(nt) + "\n")
            tree_file_rules.write("Number of trees: " + str(nt) + "\n")
            tree_file.write("Number of features: " + str(f) + "\n")
            tree_file_rules.write("Number of features: " + str(f) + "\n")
            tree_file.write("Feature importance: " + str(feat_importance) + "\n")
            tree_file_rules.write("Feature importance: " + str(feat_importance) + "\n")
            tree_file.write("Train Accuracy: " + str(acc_train) + "\n")
            tree_file_rules.write("Train Accuracy: " + str(acc_train) + "\n")
            tree_file.write("Test Accuracy: " + str(acc) + "\n")
            tree_file_rules.write("Test Accuracy: " + str(acc) + "\n")
            tree_file.write("Execution time: " + str(timedelta(seconds=end_time - start_time)) + "\n")
            tree_file_rules.write("Execution time: " + str(timedelta(seconds=end_time - start_time)) + "\n")
            tree_file.write("#-------\n")
            tree_file_rules.write("#-------\n")
            for t in range(nt):
                tree_file_rules.write("Tree " + str(t) + ": " + str(final_trees) + " \n")

            nt_vec.append(nt)
            f_vec.append(f)
            acc_vec.append(acc)
            train_acc_vec.append(acc_train)
            time_vec.append(end_time - start_time)

    df_dict = {'nt': nt_vec, 'nf': f_vec, 'train_acc': train_acc_vec, 'acc': acc_vec, 'time': time_vec}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv('../results/' + dataset['name'] + '/rf/' + dataset['name'] + '_params_rf.csv', index=False)

    for f in dataset['nf_rf']:
        plt.plot(df[df['nf'] == f]['nt'], df[df['nf'] == f]['acc'], label='NF = {}'.format(str(f)))
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../results/' + dataset['name'] + '/rf/' + dataset['name'] + '_params_rf.svg')
    plt.clf()

    for f in dataset['nf_rf']:
        plt.plot(df[df['nf'] == f]['nt'], df[df['nf'] == f]['time'], label='NF = {}'.format(str(f)))
    plt.xlabel('Number of trees')
    plt.ylabel('Execution time')
    plt.legend()
    plt.savefig('../results/' + dataset['name'] + '/rf/' + dataset['name'] + '_params_rf_time.svg')
    plt.clf()

    tree_file.close()
    tree_file_rules.close()


########
# - Decision Forest -
########
for dataset in datasets:

    print("\n###########")
    print("Data set:", dataset['name'])
    print("###########")

    tree_file_rules = open('../results/' + dataset['name'] + '/df/' + dataset['name'] + '_trees_df_rules.txt', "w")
    tree_file = open('../results/' + dataset['name'] + '/df/' + dataset['name'] + '_trees_df.txt', "w")
    nt_vec = []
    f_vec = []
    acc_vec = []
    train_acc_vec = []
    time_vec = []
    for nt in dataset['nt']:
        for f in dataset['nf_df']:
            start_time = time.monotonic()
            model = DecisionForestClassifier(dataset_name=dataset['name'], target_name=dataset['target_variable'],
                                             cat_variables=dataset['cat_variables'],
                                             test_split=0.2, n_estimators=nt, number_features=f)
            X_train, y_train, X_val, y_val = model.import_data()
            model.fit(X_train, y_train)
            final_trees = model.show_trees()
            pred = model.predict(X_val)
            acc_train, acc = model.evaluate(X_val, y_val)
            feat_importance = model.feature_importance()
            end_time = time.monotonic()

            print("Number of trees:", str(nt))
            print("Number of features:", str(f))
            print("Train accuracy:", acc_train)
            print("Test accuracy:", acc)
            print("Feature importance:", feat_importance)

            tree_file.write("\n#-------\n")
            tree_file_rules.write("\n#-------\n")
            tree_file.write("Number of trees: " + str(nt) + "\n")
            tree_file_rules.write("Number of trees: " + str(nt) + "\n")
            tree_file.write("Number of features: " + str(f) + "\n")
            tree_file_rules.write("Number of features: " + str(f) + "\n")
            tree_file.write("Feature importance: " + str(feat_importance) + "\n")
            tree_file_rules.write("Feature importance: " + str(feat_importance) + "\n")
            tree_file.write("Train Accuracy: " + str(acc_train) + "\n")
            tree_file_rules.write("Train Accuracy: " + str(acc_train) + "\n")
            tree_file.write("Test Accuracy: " + str(acc) + "\n")
            tree_file_rules.write("Test Accuracy: " + str(acc) + "\n")
            tree_file.write("Execution time: " + str(timedelta(seconds=end_time - start_time)) + "\n")
            tree_file_rules.write("Execution time: " + str(timedelta(seconds=end_time - start_time)) + "\n")
            tree_file.write("#-------\n")
            tree_file_rules.write("#-------\n")
            for t in range(nt):
                tree_file_rules.write("Tree " + str(t) + ": " + str(final_trees) + " \n")

            nt_vec.append(nt)
            f_vec.append(f)
            acc_vec.append(acc)
            train_acc_vec.append(acc_train)
            time_vec.append(end_time - start_time)

    df_dict = {'nt': nt_vec, 'nf': f_vec, 'train_acc': train_acc_vec, 'acc': acc_vec, 'time': time_vec}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv('../results/' + dataset['name'] + '/df/' + dataset['name'] + '_params_df.csv', index=False)

    for f in dataset['nf_df']:
        plt.plot(df[df['nf'] == f]['nt'], df[df['nf'] == f]['acc'], label='NF = {}'.format(str(f)))
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../results/' + dataset['name'] + '/df/' + dataset['name'] + '_params_df.svg')
    plt.clf()

    for f in dataset['nf_df']:
        plt.plot(df[df['nf'] == f]['nt'], df[df['nf'] == f]['time'], label='NF = {}'.format(str(f)))
    plt.xlabel('Number of trees')
    plt.ylabel('Execution time')
    plt.legend()
    plt.savefig('../results/' + dataset['name'] + '/df/' + dataset['name'] + '_params_df_time.svg')
    plt.clf()

    tree_file.close()
    tree_file_rules.close()

