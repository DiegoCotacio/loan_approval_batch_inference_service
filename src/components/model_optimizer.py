import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from .model_evaluator import Evaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")



class Hyperparameter_Optimization:

    """
    Class for doing hyperparameter optimization

    """

    def __init__(
              self, X_train: np.ndarray,
              y_train: np.ndarray,
              X_test: np.ndarray,
              y_test: np.ndarray,
              
              ) -> None:
          """Initialize the class with the training and test data."""
          self.X_train = X_train
          self.y_train = y_train
          self.X_test = X_test
          self.y_test = y_test

    def classifier_optimizer(self, trial: optuna.Trial) -> float:
        
        try:
             param = {
                  "max_depth": trial.suggest_int("max_depth", 5, 30),
                  "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 10.0),
                  "n_estimators": trial.suggest_int("n_estimators", 1, 300),
                  "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                }
             
             reg = LGBMClassifier(**param)
             reg.fit(self.X_train, self.y_train)
             val_accuracy = reg.score(self.X_test, self.y_test)

             return val_accuracy

        except Exception as e:
            raise CustomException(e, sys)
