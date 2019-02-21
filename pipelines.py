#%% Pipelines
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics 
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as randint
from scipy.stats import uniform
from os import mkdir

# Set Up Data
print('Splitting Data into Training and Validation Sets')
X_train, X_test, y_train, y_test = train_test_split(data['raw_text'], 
                data['target'], train_size=0.8, test_size=0.2, random_state=0)

# Utility Function to Report Best Scores
def report(results, n_top, trial, runtime, path):
    
    filename = "./src/models/" + path + "/test_report_" + str(trial) + ".txt"
    f = open(filename,"a")
    f.write("Trial, Parameters, Max Validation Score, St Dev, Time\n")
    
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        top_candidate = np.flatnonzero(results['rank_test_score'] == 1)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
            f.write(trial + ", " + 
                    str(results['params'][candidate]) + ", " +
                    str(results['mean_test_score'][candidate]) + ", " +
                    str(results['std_test_score'][candidate]) + ", " +
                    str(runtime) + "\n")
            
    f.close()
    return results['params'][top_candidate[0]]

## Feature Extraction Pipelines
# Binary Occurance/Logistic Regression
pipe_bo_lr = Pipeline([('vect', CountVectorizer() ),
                       ('norm', Normalizer() ),
                       ('clf', LogisticRegression() )])

# TF-IDF/Logistic Regression
pipe_tfidf_lr = Pipeline([('vect', CountVectorizer() ),
                          ('tfidf', TfidfTransformer() ),
                          ('norm', Normalizer() ),
                          ('clf', LogisticRegression() )]) 

# Binary Occurance/Decision Tree
pipe_bo_dt = Pipeline([('vect', CountVectorizer() ),
                       ('norm', Normalizer() ),
                       ('clf', DecisionTreeClassifier() )])

# TF-IDF/Logistic Regression
pipe_tfidf_dt = Pipeline([('vect', CountVectorizer() ),
                          ('tfidf', TfidfTransformer() ),
                          ('norm', Normalizer() ),
                          ('clf', DecisionTreeClassifier() )]) 

pipeline_list = {#"Pipe_1_BO-LR": pipe_bo_lr
#                 "Pipe_2_TFIDF-LR": pipe_tfidf_lr, 
#                 "Pipe_3_BO-DT": pipe_bo_dt, 
#                 "Pipe_4_TFIDF-DT": pipe_tfidf_dt
#                 "Pipe_5_BO-NGRAM-LR": pipe_bo_lr
#                 "Pipe_6_TRUE_BO-NGRAM-LR": pipe_tfidf_lr,
#                  "Pipe_7_BO-TFIDF-LR": pipe_tfidf_lr}
                  #"Pipe_8_BO-TFIDF-LR": pipe_tfidf_lr}
                  "Pipe_11_BO-TFIDF-LR": pipe_tfidf_lr}
  
print('Defining Parameters')

#tfidf_params = {"vect__ngram_range": [(1,1)], 
#                "tfidf__use_idf": [True]}
#
#dt_params = {"clf__min_impurity_decrease": [1e-5], 
#             "clf__min_samples_split": [2,3,4]}
#
#lr_params = {"clf__tol":[1e-4], "clf__solver":["lbfgs"]}

bo_params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)],
                                   "vect__binary": [True,False]}

tfidf_params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)], 
                "tfidf__use_idf": [True,False]}

dt_params = {"clf__min_impurity_decrease": [1e-5,5e-6,2e-6,1e-6], 
             "clf__min_samples_split": [2,5,10]}

lr_params = {"clf__tol":[1e-4,1e-5,5e-6,1e-6], "clf__solver":["lbfgs"]}


bo_lr_params = lr_params.copy()
tfidf_lr_params = lr_params.copy()

bo_dt_params = dt_params.copy()
tfidf_dt_params = dt_params.copy()

bo_lr_params.update(bo_params)
bo_dt_params.update(bo_params)

tfidf_lr_params.update(tfidf_params)
tfidf_dt_params.update(tfidf_params)

bo_tfidf_lr_params = {"vect__binary": [True], 
                      "vect__ngram_range": [(1,2)],
                      "tfidf__use_idf": [True],
                      "clf__solver": ['lbfgs'],
                      "clf__C": [1,10,100,1000]}

bo_tfidf_lr_params_2 = {"vect__binary": [True], 
                      "vect__ngram_range": [(1,2)],
                      "tfidf__use_idf": [True],
                      "clf__solver": ['lbfgs'],
                      "clf__C": [2000,5000,10000,20000,50000]}

bo_tfidf_lr_params_3 = {"vect__binary": [True], 
                      "vect__ngram_range": [(1,2)],
                      "tfidf__use_idf": [True],
                      "clf__solver": ['lbfgs'],
                      "clf__C": [10000],
                      "clf__max_iter": [100,150,200,250,300] }


bo_tfidf_lr_params_4 = {"vect__binary": [True], 
                      "vect__ngram_range": [(1,2)],
                      "tfidf__use_idf": [True],
                      "clf__solver": ['lbfgs'],
                      "clf__C": [8600,8700,8800,8900,9100,9200,9300,9400],
                      "clf__max_iter": [150] }

param_master_grid = {#"Pipe_1_BO-LR": lr_params,
                     #"Pipe_2_TFIDF-LR": tfidf_lr_params,
                     #"Pipe_3_BO-DT": bo_dt_params,
                     #"Pipe_4_TFIDF-DT": tfidf_dt_params
                     #"Pipe_6_TRUE_BO-NGRAM-LR": bo_lr_params
                     "Pipe_11_BO-TFIDF-LR": bo_tfidf_lr_params_4}


#%% Run grid search  


path = "run_" + time.strftime("%m%d-%H%M")
mkdir("./src/models/{}".format(path))

for pipe in pipeline_list:
    print('Performing Grid Search for {}'.format(pipe))
    grid_search = GridSearchCV(pipeline_list[pipe], 
                               param_grid = param_master_grid[pipe], 
                               cv=5)
                                #cv=5)
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    end = time.time()
    
    y_pred = grid_search.predict(X_test)
    runtime = end - start
    print("GridSearchCV for {} took {} seconds".format(pipe,runtime))
    
    top_params = report(grid_search.cv_results_, 5, pipe, runtime, path)
    
    cl_rpt = metrics.classification_report(y_test,y_pred, target_names=('Negative','Positive'))
    print(cl_rpt)
    
    filename = "./src/models/{}/validation_test_report_{}".format(path,pipe) + ".txt"
    f = open(filename,'a')
    f.write(str(top_params) + "\n")
    f.write(cl_rpt)
    f.close()
    

# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
