from sklearn.metrics import f1_score

def calculate_accuracy(output, target):
    acc = ((output.argmax(dim=1) == target).float().mean())
    return acc


def model_f1_score(y, y_pred):
    return f1_score(y.cpu().data.max(1)[1], y_pred.cpu())
