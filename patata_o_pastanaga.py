from keras.models import load_model, model_from_json
from keras.preprocessing import image
import numpy as np
import sys

if len(sys.argv) != 1:
    FILE = sys.argv[1]
else:
    FILE = "test.jpg"


def potato_or_carrot(model, x):
    pred = model.predict(x)
    if pred[0] > 0.5:
        print("Te pinta de que es una patata, " + str(pred[0][0]*100) + "% de confiança")
    else:
        print("Te pinta de que es una pastanaga, " + str((1 - pred[0][0])*100) + "% de confiança")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

img = image.load_img(FILE, target_size = (64,64))
img_to_show = image.load_img(FILE)
img_to_show.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

print("\n\n\n\n")
potato_or_carrot(loaded_model, x)
