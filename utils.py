from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
import numpy as np

def score_CI(model, X, y):
    D, T = map(np.array, zip(*y))
    prediction = model.predict(X)
    result = concordance_index_censored(D.astype('?'), T, prediction)
    return result[0]

def score_CI_ipcw(model, X_test, Y_train, Y_test):
    Y_censor = np.concatenate((Y_train, Y_test))
    estimate = model.predict(X_test)
    _, T_train = map(np.array, zip(*Y_train))
    tau = np.sort(T_train)[-2]
    return concordance_index_ipcw(Y_censor, Y_test, estimate, tau=tau)[0]

def score_brier(model, X_test, Y_train, Y_test):
    Y_censor = np.concatenate((Y_train, Y_test))

    predicted_curves = model.predict_survival_function(X_test)
    D_train, T_train = map(np.array, zip(*Y_train))
    D_test, T_test = map(np.array, zip(*Y_test))
    T_test = np.unique(np.sort(T_test[D_test]))

    min = T_train[D_train].min()
    max = T_train[D_train].max()
    T_test =  T_test[np.logical_and(T_test > min, T_test < max)]
    T_test = T_test[1:-1]
    preds = np.asarray([[fn(t) for t in T_test] for fn in predicted_curves])
    preds = np.nan_to_num(preds)
    try:
        return integrated_brier_score(Y_censor, Y_test, preds, T_test)
    except:
        return 0.5