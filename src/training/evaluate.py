import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_labels(generator):
    labels = [None] * len(generator.class_indices)
    for label, index in generator.class_indices.items():
        labels[index] = label
    return labels


def build_evaluation_generator(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=shuffle,
    )
    return generator


def compute_classification_metrics(y_true, y_pred, average_methods=None):
    if average_methods is None:
        average_methods = ['macro', 'weighted']
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    return metrics


def compute_per_class_accuracy(conf_matrix, labels):
    row_sums = conf_matrix.sum(axis=1).astype(float)
    per_class = {}
    for idx, label in enumerate(labels):
        if row_sums[idx] > 0:
            per_class[label] = float(conf_matrix[idx, idx] / row_sums[idx])
        else:
            per_class[label] = 0.0
    return per_class


def compute_roc_auc(y_true, probabilities, labels):
    n_classes = len(labels)
    if n_classes < 2:
        return {'roc_auc': None}

    y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
    if probabilities.shape[1] != n_classes:
        raise ValueError('Probability vector length does not match number of classes.')

    try:
        macro_roc_auc = float(roc_auc_score(y_true_binarized, probabilities, average='macro', multi_class='ovr'))
        weighted_roc_auc = float(roc_auc_score(y_true_binarized, probabilities, average='weighted', multi_class='ovr'))
    except Exception:
        macro_roc_auc = None
        weighted_roc_auc = None
    return {'roc_auc_macro': macro_roc_auc, 'roc_auc_weighted': weighted_roc_auc}


def get_misclassified_examples(generator, y_true, y_pred, probabilities, top_n=10):
    examples = []
    filepaths = getattr(generator, 'filepaths', None)
    labels = get_labels(generator)
    misclassified_idx = np.where(y_true != y_pred)[0]

    for idx in misclassified_idx[:top_n]:
        example = {
            'true_label': labels[int(y_true[idx])],
            'predicted_label': labels[int(y_pred[idx])],
            'confidence': float(np.max(probabilities[idx])),
        }
        if filepaths is not None:
            example['filepath'] = filepaths[int(idx)]
        examples.append(example)
    return examples


def compute_confidence_distribution(probabilities, bins=10):
    scores = np.max(probabilities, axis=1)
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    return counts.tolist(), edges.tolist()


def plot_confusion_matrix(conf_matrix, labels):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    thresh = conf_matrix.max() / 2.0 if conf_matrix.max() else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                int(conf_matrix[i, j]),
                ha='center',
                va='center',
                color='white' if conf_matrix[i, j] > thresh else 'black',
            )

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    return fig


def plot_confidence_distribution(counts, edges):
    fig, ax = plt.subplots(figsize=(8, 4))
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
    ax.bar(centers, counts, width=edges[1] - edges[0], edgecolor='black')
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Number of examples')
    fig.tight_layout()
    return fig


def evaluate_generator(model, generator, top_n_misclassified=10, bins=10):
    probabilities = model.predict(generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=-1)
    y_true = generator.classes
    labels = get_labels(generator)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update(compute_roc_auc(y_true, probabilities, labels))
    cm = confusion_matrix(y_true, y_pred)
    metrics['per_class_accuracy'] = compute_per_class_accuracy(cm, labels)
    metrics['classification_report'] = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    metrics['confusion_matrix'] = cm
    metrics['labels'] = labels
    metrics['misclassified_examples'] = get_misclassified_examples(generator, y_true, y_pred, probabilities, top_n=top_n_misclassified)
    metrics['confidence_distribution'] = compute_confidence_distribution(probabilities, bins=bins)
    return metrics
