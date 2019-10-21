## Steps

+ *Normalize* the Dataset
+ *Build* the Model
+ *Train* the model

## Prepare the Datasets

```py
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## Define the Structure

```py
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])
```

1. `Flatten` transforms the 2D images (28px x 28px) to a a 1D Array (of size 28 * 28)
2. 1st layer has 128 nodes (relu)
3. 2nd layer has 10 nodes (softmax)

## Configure the Model

```py
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
```

+ **optimizer**: Define training procedure - how the model is updated
+ **loss**: minimization function used in optimization
+ **metrics**: used to monitor training and testing steps

## Train the Model

```py
model.fit(train_images, train_labels, epochs=10)
```

+ Feeds training data to model. The model learns to associate images with labels
+ epoch: iteration over the entire input data.

## Evaluate the Accuracy

Compare how models pferoms on test dataset

```py
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

## Make Predictions

Apply trained model on new datasets

```py
predictions = model.predict(test_images)
```
