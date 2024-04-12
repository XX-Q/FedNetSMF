import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate_MAE(m1, m2):
    return mean_absolute_error(m1, m2)

def calculate_MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def evaluate_LINE():
    m1 = np.load("evaluation/line_embedding_1000.npy")
    m1 = np.abs(m1)
    m2 = np.load("evaluation/fed_line_embedding_1000.npy")

    m2 = np.abs(m2)
    MAPE = calculate_MAPE(m1,m2)
    print("LINE MAPE:",MAPE)

def evaluate_DeepWalk():
    m1 = np.load("evaluation/deepwalk_embedding_1000.npy")
    m1 = np.abs(m1)
    m2 = np.load("evaluation/fed_deepwalk_embedding_1000.npy")

    m2 = np.abs(m2)
    MAPE = calculate_MAPE(m1,m2)
    print("DeepWalk MAPE:",MAPE)

if __name__ == '__main__':
    evaluate_LINE()
    evaluate_DeepWalk()
    # print(1)