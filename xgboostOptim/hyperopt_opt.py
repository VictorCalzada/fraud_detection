import numpy as np
import xgboost as xgb
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, space_eval, tpe
from xgboost import XGBClassifier
from hyperopt import hp
from sklearn.metrics import accuracy_score, roc_curve



class HPOpt(object):
    def __init__(self, X_train, X_test, y_train, y_test, best_threshold=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_threshold = best_threshold
        self.model = XGBClassifier()
        self.space = {
            "learning_rate": hp.choice("learning_rate", np.arange(0.02, 0.9, 0.05)),
            "max_depth": hp.choice("max_depth", np.arange(5, 40, 1, dtype=int)),
            "min_child_weight": hp.choice("min_child_weight", np.arange(0.001, 0.9, 0.1)),
            "min_child_samples": hp.choice("min_child_samples", np.arange(2, 100, 1, dtype=int)),
            "colsample_bytree": hp.choice("colsample_bytree", np.arange(0.1, 0.9, 0.1)),
            "subsample": hp.choice("subsample", np.arange(0.1, 1, 0.05)),
            "n_estimators": hp.choice("n_estimators", np.arange(50, 500, 5, dtype=int)),
            "num_leaves": hp.choice("num_leaves", np.arange(10, 150, 15, dtype=int)),
            "reg_alpha": hp.choice("reg_alpha", np.arange(0.1, 1, 0.1)),
            "reg_lambda": hp.choice("reg_lambda", np.arange(0.5, 1.5, 0.1)),
            "objective": "binary"
        }

        '''
        xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
        '''


    def fit(self, max_evals):
        trials = Trials()
        fn = getattr(self, "fit_")
        try:
            result = fmin(
                fn=fn,
                space=self.space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(1),
            )
        except Exception as e:
            return {"status": STATUS_FAIL, "exception": str(e)}
        opt_params = space_eval(self.space, result)
        return opt_params

    def fit_(self, para):
        reg = self.model.set_params(**para)
        reg.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            callbacks = [
                lgb.early_stopping(10, verbose=0), 
                lgb.log_evaluation(period=0)
            ]
        )
        probs = reg.predict_proba(self.X_test)[:,1]
        if self.best_threshold:
            fpr, tpr, thresholds = roc_curve(self.y_test, probs, pos_label=1)
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            threshold = thresholds[ix]
        else:
            threshold = 0.5
        y_pred = np.where(probs >= threshold, 1, 0)
        accuracy = accuracy_score(y_pred, self.y_test)
        loss = 1 - accuracy
        return {"loss": loss, "status": STATUS_OK}
