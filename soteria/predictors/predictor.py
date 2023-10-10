# _*_ coding: utf-8 _*_
from .carts import CART


def fit_acc_predictor(inputs, targets):
    model = CART()
    model.fit(inputs, targets)
    return model, model.predict(inputs)
