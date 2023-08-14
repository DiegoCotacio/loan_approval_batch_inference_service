import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation:

    def __init__(self) -> None:
        pass

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the accuracy method of the Evaluation Class")
            acc = accuracy_score(y_true, y_pred)
            logging.info("The accuracy value is: " + str(acc))
            return acc
        except Exception as e:
            logging.info("Exception occurred in the accuracy method of the Evaluation class. Exception message: " + str(e))
            raise Exception()

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the precision method of the Evaluation Class")
            prec = precision_score(y_true, y_pred)
            logging.info("The precision value is: " + str(prec))
            return prec
        except Exception as e:
            logging.info("Exception occurred in the precision method of the Evaluation class. Exception message: " + str(e))
            raise Exception()

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the recall method of the Evaluation Class")
            rec = recall_score(y_true, y_pred)
            logging.info("The recall value is: " + str(rec))
            return rec
        except Exception as e:
            logging.info("Exception occurred in the recall method of the Evaluation class. Exception message: " + str(e))
            raise Exception()

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the f1_score method of the Evaluation Class")
            f1 = f1_score(y_true, y_pred)
            logging.info("The f1_score value is: " + str(f1))
            return f1
        except Exception as e:
            logging.info("Exception occurred in the f1_score method of the Evaluation class. Exception message: " + str(e))
            raise Exception()