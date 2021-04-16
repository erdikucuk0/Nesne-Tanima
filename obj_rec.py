# libraries that we need
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# load the earlier trained model
model = ResNet50(weights="imagenet")


# image must be 224*224 dimensions, created a function to reshape it
def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

# here you should paste url of an image
# i chose a glasses img, you can try with whatever you want :)
ImageURL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQd7SolZCs6-o2AfdJxD7q2C9vycgGZlBPTmg&usqp=CAU"
response = requests.get(ImageURL)
image = Image.open(BytesIO(response.content))
data = {"success": False}

# reshape 224*224
prepared_image = prepare_image(image, target=(224, 224))
preds = model.predict(prepared_image)

results = imagenet_utils.decode_predictions(preds)
data["predictions"] = []

for (imagenetID, label, prob) in results[0]:
    r = {"label": label, "probability": float(prob)}
    data["predictions"].append(r)

data["success"] = True

# prediction result
print("The classification estimate is {0} with the highest ratio of {1}. ".format(data["predictions"][0]["label"],
                                                                            data["predictions"][0]["probability"]))
