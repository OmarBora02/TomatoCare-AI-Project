from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classfier = PredictionPipeline(self.filename)

    
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.files['file']
    image_path = "inputImage.jpg"  
    image.save(image_path)

    client_app = ClientApp()
    result = client_app.classfier.predict()  

    return jsonify({"prediction": result[0]["image"]})




if __name__ == "__main__":
    client_app = ClientApp()
    app.run(host='0.0.0.0', port=8080)