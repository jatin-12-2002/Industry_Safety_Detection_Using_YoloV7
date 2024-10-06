import sys,os,shutil
from isd.pipeline.training_pipeline import TrainPipeline
from isd.exception import isdException
from isd.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from isd.configuration.s3_operations import S3Operation
from isd.entity.config_entity import ModelPusherConfig

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.s3 = S3Operation()  # Create S3Operation instance
        self.filename = "data/inputImage.jpg"

        self.model_path = 'yolov7/best.pt'
        self.bucket_name = self.model_pusher_config.MODEL_BUCKET_NAME

        # Ensure the 'yolov7/' directory exists
        if not os.path.exists('yolov7'):
            os.makedirs('yolov7')

        # Check if model exists locally, if not, download it from S3 or start training
        if not os.path.exists(self.model_path):
            if self.s3.is_model_present(self.bucket_name, 'best.pt'):
                self.s3.download_object(key=self.model_pusher_config.S3_MODEL_KEY_PATH, 
                                        bucket_name=self.bucket_name, filename=self.model_path)
            else:
                print("Model not found in S3, starting training process...")
                # Use Flask's app context to call the train route directly
                with app.app_context():
                    trainRoute()  # Call the training route function directly
                # Check again if the model is now available after training
                if not os.path.exists(self.model_path):
                    raise Exception("Training completed, but model was not saved successfully.")


@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()  # This will train the model and save it
        print("Training completed successfully!")
        return "Training completed successfully!"
    except Exception as e:
        return Response(f"Error during training: {str(e)}", status=500)
    


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        # Check if the image is in the request
        if 'image' not in request.json:
            return Response("No image file found in the request.", status=400)

        # Ensure the "data" directory exists
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Decode and save the image to the expected location
        image = request.json['image']
        decodeImage(image, clApp.filename)

        if not os.path.exists(clApp.filename):
            return Response("Failed to save the image", status=400)
       
        os.system("cd yolov7/ && python detect.py --weights best.pt  --source ../data/inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64("yolov7/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}

        output_dir="yolov7/runs"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Delete the entire 'runs' folder

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data", status=400)
    except KeyError:
        return Response("Key value error: incorrect key passed", status=400)
    except Exception as e:
        print(e)
        return Response(f"An error occurred during prediction: {str(e)}", status=500)

    return jsonify(result)


if __name__ == "__main__":
    # Initialize the ClientApp instance with correct arguments (this needs to be passed in correctly)
    model_pusher_config = ModelPusherConfig()  # Instantiate with the correct configuration
    clApp = ClientApp(model_pusher_config=model_pusher_config)
    app.run(host="0.0.0.0", port=8080)