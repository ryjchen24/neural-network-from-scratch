from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data.values, mnist.target.values.astype(int)

# Must do because initially each pixel gets a lightness/darkness value from 0-255 so this makes it a range 0-1
X = X / 255.0

one_hot_y = np.zeros((y.shape[0], 10))
one_hot_y[np.arange(y.shape[0]), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, one_hot_y, test_size=0.2, random_state=67)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=67)

nn = NeuralNetwork([784, 128, 64, 10])

train_losses, val_accuracies = nn.train_model(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=15,
    batch_size=32,
    lr=0.01
)

final_test_accuracy = nn.evaluate_accuracy(X_test, y_test)
print(f"Final test accuracy: {final_test_accuracy:.4f}")




# Visualization of results

# Plot metrics
fig, (g1, g2) = plt.subplots(1, 2, figsize=(12, 6))

g1.plot(train_losses)
g1.set_xlabel("Epoch")
g1.set_ylabel("Loss")
g1.set_title("Training Loss by Epoch")
g1.grid(True)

g2.plot(val_accuracies)
g2.set_xlabel("Epoch")
g2.set_ylabel("Accuracy")
g2.set_title("Validation Accuracy by Epoch")
g2.grid(True)

plt.tight_layout()
plt.show()


# Confusion Matrix
y_pred = nn.predict(X_test)
y_true = np.argmax(y_test, axis=1)
class_names = [str(i) for i in range(10)]

confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
fig, ax = plot_confusion_matrix(
    conf_mat=confmat,
    class_names=class_names,
    figsize=(12, 8)
)
plt.xticks(rotation=0)
plt.show()


# Example predictions vs true values
fig, subplots = plt.subplots(4, 5, figsize=(12, 6))
indices = np.random.choice(X_test.shape[0], 20)

# Flat since it is a 2d plot but we can change to 1d so can iterate through it
for i, axes in enumerate(subplots.flat):
    img = X_test[indices[i]].reshape(28, 28)
    pred_label = nn.predict(X_test[indices[i]:indices[i]+1])[0]
    true_label = np.argmax(y_test[indices[i]])
    title_text = f"Pred: {pred_label} | True: {true_label}"

    axes.imshow(img, cmap="gray")
    if pred_label == true_label:
      axes.set_title(title_text, fontsize=10, c="g", pad=5) # green text if correct
    else:
      axes.set_title(title_text, fontsize=10, c="r", pad=5) # red text if wrong
    axes.axis(False)
plt.show()