# Нужные библиотеки
from sklearn.metrics import f1_score, confusion_matrix

# Точность
def calculate_accuracy(output, target):
    acc = ((output.argmax(dim=1) == target).float().mean())
    return acc

# F1-score
def model_f1_score(y, y_pred):
    return f1_score(y.cpu().data.max(1)[1], y_pred.cpu())

# Матрица ошибок
def model_matrix(y, y_pred):
    return confusion_matrix(y.data.max(1)[1], y_pred.cpu())
