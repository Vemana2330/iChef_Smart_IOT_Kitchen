import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class Predictor:
    @staticmethod
    def predict_image(model, image_path, target_size=(150, 150)):
        img = image.load_img(image_path, target_size=target_size)
        x = image.img_to_array(img)
        plt.imshow(x / 255.0)
        plt.show()
        
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        predictions = model.predict(x)
        
        if predictions[0] < 0.5:
            return "Burnt Food"
        else:
            return "Cooked Food"
