# Import libraries for evaluation of the models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to evaluate the model performance
def evaluate_model(model_name, true_labels, predicted_labels, class_labels):

    # Calculate metrics 
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Annotate the confusion matrix with counts
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.tight_layout()
    plt.show()

    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# Function to print a summary table of the metrics
def summarize_metrics(metrics_dict):

    # Define the table header with column names
    header = f"{'Model':<25}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}"
    print("\n",header)
    print("-" * len(header))
    
    # Print each row of the table with the results from the dictionary of metrics
    for model_name, metrics in metrics_dict.items():
        row = f"{model_name:<25}{metrics['accuracy']:<12.4f}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}{metrics['f1_score']:<12.4f}"
        print(row)