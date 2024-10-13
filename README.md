# Fashion MNIST Classification using Convolutional Neural Networks (CNNs)

In this project, we develop deep learning models to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The dataset contains 28x28 grayscale images of various clothing items. Our goal is to build two Convolutional Neural Networks (CNNs) to accurately classify these images into one of 10 categories, such as T-shirts, sneakers, and dresses. We'll start with a simple CNN model and then improve it using advanced techniques like Batch Normalization and Dropout.

## Objectives

- Implement a baseline CNN model for Fashion MNIST classification.
- Enhance the model performance by incorporating Batch Normalization, Dropout, and additional convolutional layers.
- Evaluate the models on training, validation, and test sets.
- Visualize model performance in terms of loss and accuracy.
- Compare the accuracy of both models.

## Dataset

The Fashion MNIST dataset consists of:
- **60,000 training images** and **10,000 test images**.
- Each image is a 28x28 pixel grayscale image, classified into one of 10 categories:

| Label | Category      |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

![1](https://github.com/user-attachments/assets/cd614dbd-5d07-462f-b34b-f98e13ee876d)

## CNN 1 Architecture

The first CNN consist two convolutional layers, max pooling, and a fully connected layer. The architecture is as follows:

### Model Architecture:

1. **Input:** 28x28 grayscale image
2. **Conv Layer 1:** 32 filters, 3x3 kernel, followed by ReLU activation and 2x2 max pooling
3. **Conv Layer 2:** 64 filters, 3x3 kernel, followed by ReLU activation and 2x2 max pooling
4. **Fully Connected Layer:** 128 units with ReLU activation
5. **Output Layer:** 10 units with softmax for classification

### Training:

- Optimizer: Adam
- Loss Function: CrossEntropy
- Epochs: 50
- Learning Rate: 0.001

### Results:

- **Test Accuracy:** 90.1%
- Loss and accuracy curves were plotted over the training epochs to monitor the performance.

---

## CNN 2 Architecture

The second model enhances the baseline CNN by adding Batch Normalization, Dropout, and an additional convolutional layer. These techniques improve training stability and reduce overfitting.

1. **Conv Layer 1:** 32 filters, BatchNorm, ReLU, 2x2 max pooling, Dropout (25%)
2. **Conv Layer 2:** 64 filters, BatchNorm, ReLU, 2x2 max pooling, Dropout (25%)
3. **Conv Layer 3:** 128 filters, BatchNorm, ReLU, 2x2 max pooling, Dropout (25%)
4. **Fully Connected Layer:** 128 units with ReLU, Dropout (50%)
5. **Output Layer:** 10 units with softmax

### Training:

- Optimizer: Adam
- Loss Function: CrossEntropy
- Epochs: 50
- Learning Rate: 0.0001

### Results:

- **Test Accuracy:** 91.37%
- This model achieves a slight improvement over the baseline, attributed to Batch Normalization and Dropout.

---

## Performance Evaluation

Both models were trained and tested on the Fashion MNIST dataset, and their performance was evaluated based on test accuracy.

| Model   | Test Accuracy |
|---------|---------------|
| CNN     | 90.1%         |
| CNN2    | 91.37%        |

The improved model (CNN2) shows better generalization, with a significant boost in performance due to the addition of BatchNorm and Dropout layers.

---

## Visualizations

During training, we monitored the performance by plotting:
- Loss vs Epochs
- Accuracy vs Epochs

These visualizations helped us understand how both models performed during training and validation.

---

## Conclusion

- The **first CNN model** provides a solid baseline for Fashion MNIST classification, achieving reasonable accuracy with a simple architecture.
- The **second CNN model** boosts performance by reducing overfitting and stabilizing the training process through Batch Normalization and Dropout.

### Future Work:
- Exploring more advanced architectures like ResNets or VGGs.
- Hyperparameter tuning, such as adjusting the learning rate, dropout rates, and number of layers.
- Fine-tuning on similar datasets to improve model generalization.

---

## References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---
