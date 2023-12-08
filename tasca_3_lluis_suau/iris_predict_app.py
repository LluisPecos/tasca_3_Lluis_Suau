import pickle
from flask import Flask, jsonify, request
from iris_predict_service import predict_single

app = Flask('iris-predict')
class_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Logistic regression
with open('models/logistic-regression-model.pck', 'rb') as f:
    logistic_regression_sc, logistic_regression_model = pickle.load(f)

# SVM, Support Vector Machine
with open('models/svm-model.pck', 'rb') as f:
    svm_sc, svm_model = pickle.load(f)

# Decision Trees
with open('models/decision-trees-model.pck', 'rb') as f:
    decision_trees_sc, decision_trees_model = pickle.load(f)

# KNN, k-Nearest Neighbours
with open('models/knn-model.pck', 'rb') as f:
    knn_sc, knn_model = pickle.load(f)

@app.route('/logistic-regression-predict', methods=['POST'])
@app.route('/svm-predict', methods=['POST'])
@app.route('/decision-trees-predict', methods=['POST'])
@app.route('/knn-predict', methods=['POST'])
def predict():
    current_route = request.path
    sc = None
    model = None

    if(current_route == '/logistic-regression-predict'):
        sc = logistic_regression_sc
        model = logistic_regression_model
        
    elif(current_route == '/svm-predict'):
        sc = svm_sc
        model = svm_model

    elif(current_route == '/decision-trees-predict'):
        sc = decision_trees_sc
        model = decision_trees_model

    elif(current_route == '/knn-predict'):
        sc = knn_sc
        model = knn_model

    customer = request.get_json()
    prediction = predict_single(customer, sc, model)

    result = {
        'iris-class': int(prediction[0]),
        'name': class_names[prediction[0]]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)