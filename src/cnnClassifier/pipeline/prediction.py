import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_names = ['Bacteria_spot', 'Early_blight', 'Healthy', 'Late_blight', 'Leaf_mold', 
                            'Mosaic_virus', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
                            'Target_spot', 'Yellow_leaf_curl_virus'] 
    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0


        # Make prediction
        result = np.argmax(model.predict(test_image), axis=1)[0]
        prediction = self.class_names[result]
        
        return [{"image": prediction}]