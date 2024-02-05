def calculate_accuracy(output, target):
    acc = ((output.argmax(dim=1) == target).float().mean())
    return acc
