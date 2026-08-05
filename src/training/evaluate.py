from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, data_generator):
    predictions = model.predict(data_generator)
    predicted_classes = predictions.argmax(axis=-1)
    true_classes = data_generator.classes

    report = classification_report(true_classes, predicted_classes, target_names=list(data_generator.class_indices.keys()))
    matrix = confusion_matrix(true_classes, predicted_classes)
    return report, matrix
