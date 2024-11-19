from tensorflow import keras

class InstaChefModel:
    @staticmethod
    def build_model(input_shape=(150, 150, 3)):
        model = keras.Sequential()
        
        # Convolutional and MaxPooling Layers
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPool2D(2, 2))
        
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPool2D(2, 2))
        
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPool2D(2, 2))
        
        # Flattening and Dense Layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # Binary classification
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
