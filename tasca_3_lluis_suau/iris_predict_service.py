def predict_single(customer, sc, model):
    customer = sc.transform([list(customer.values())])
    y_pred = model.predict(customer)
    return y_pred