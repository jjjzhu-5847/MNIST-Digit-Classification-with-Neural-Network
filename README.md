# 🧮 MNIST Digit Classification with Neural Networks

## 📚 Project Overview

This project demonstrates the classification of handwritten digits (0–9) from the MNIST dataset using a simple feed-forward neural network built with TensorFlow and Keras. The MNIST dataset is a classic benchmark in machine learning, and this notebook walks through the full pipeline: loading the data, preprocessing, building and training the model, evaluating performance, and making predictions—including on custom images.

---

## 🚀 Features

- 🔢 **Dataset:** Uses the MNIST dataset of 28x28 grayscale digit images
- 🛠️ **Preprocessing:** Normalizes pixel values to [0, 1] for better training performance
- 🧠 **Neural Network Design:** 
    - Input layer (flattened 28x28)
    - Two hidden layers (32 and 16 neurons, ReLU activation)
    - Output layer (10 neurons, softmax activation)
- 📊 **Visualization:** 
    - Training & validation accuracy/loss curves
    - Confusion matrix heatmap for model diagnostics
- 📝 **Evaluation:** 
    - Prints detailed metrics on the test set
    - Supports prediction on new (custom) images

---

## 🏗️ Model Architecture

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
```

- **Optimizer:** Adam
- **Loss Function:** Sparse categorical crossentropy
- **Metrics:** Accuracy

---

## 🧪 Training Summary

| Metric                 | Value (Approx.) |
|------------------------|-----------------|
| Training Accuracy      | 98%             |
| Validation Accuracy    | 98%             |
| Test Accuracy          | 96%             |
| Epochs                 | 10              |

- Training and validation accuracy both improve steadily
- No significant overfitting observed

---

## 📉 Performance Charts

**Accuracy Curve:**
<img width="619" height="449" alt="image" src="https://github.com/user-attachments/assets/df04858b-45c3-4238-8f51-fca3ca51d860" />


**Loss Curve:**
<img width="602" height="448" alt="image" src="https://github.com/user-attachments/assets/bdc83322-b85b-4e43-8612-de0ee96f1e32" />


**Confusion Matrix:**
<img width="906" height="487" alt="image" src="https://github.com/user-attachments/assets/f083e0aa-7993-4b0c-bbfd-01ee20c21c09" />
---

## 📦 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/jjjzhu-5847/MNIST-Digit-Classification-with-Neural-Network.git
    ```
2. Open the notebook `MNIST_Digit_Classification_with_Neural_Network.ipynb` in Jupyter or Google Colab.
3. Run all cells to train the model and visualize results.

---

## 🖼️ Predict Your Own Digits!

- The notebook includes code to load your own image, preprocess it, and predict the digit.
- Make sure your image is a clear 28x28 grayscale digit for best results.

---

## 📖 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)

---

## 🙌 Acknowledgments

- Developed by [Your Name]  
- Inspired by classic MNIST neural network tutorials

---

Feel free to open an issue or pull request if you have suggestions or improvements!
