# import os
from flask import Flask, jsonify,request 
# from werkzeug.utils import secure_filename

from torch import *
# from torch.utils.data import Dataset, DataLoader
# import torchvision
from torchvision import *
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
# import psycopg2
# from psycopg2 import Error
# import base64
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# GLOBAL VARIABLES:
app.config['UPLOAD_FOLDER'] = './Uploads'
birds = ('BLACK-HEADED IBIS', 'COMMON HAWK-CUCKOO', 'COMMON WOOD PIGEON', 'COPPERSMITH BARBET', 'GREAT INDIAN BUSTARD', 'GREATER COUCAL', 'GREATER FLAMINGO', 'GREEN BEE EATER', 'GREY-FRONTED GREEN PIGEON', 'INDIAN PEAFOWL', 'PAINTED STORK', 'ROCK DOVE', 'SPECKLED WOOD PIGEON', 'WHITE RUMPED VULTURE')
CLASS_LABELS = 15
torch.manual_seed(2022)
device = 'cpu'
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Download the model
model_t_s = models.inception_v3(pretrained=True)
# Update the fcc
model_t_s.fc = nn.Sequential(
nn.Linear(model_t_s.fc.in_features, 1024),
nn.Linear(1024, CLASS_LABELS)
)
# Load the learned weights
model_t_s.eval()
model_t_s.load_state_dict(torch.load('./inception_model.pt', map_location=torch.device('cpu')))
model_t_s.eval()
# Send to GPU
model_t_s.to(device)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  

# ROUTES:
@app.route('/')
@cross_origin()
def index():
  return "Server is live ..."

@app.route('/predict',methods=['POST'])
@cross_origin()
def post():
  # Check if post request has a file part

  print(dict(request.files))
  # print(dict(request.form))
  if 'file' not in request.files: 
    response = jsonify({'message': "No file part in the request"})
    response.status_code = 400
    return response

  file = request.files['file']
  if file.filename ==  '':
    response = jsonify({'message':'No file selected for uploading'})
    response.status_code = 400
    return response
  
  if file and allowed_file(file.filename):
    # filename = secure_filename(file.filename)
    img = Image.open(file)

    # Prepare the dataset and labels
    transform = transforms.Compose([transforms.Resize((299, 299)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    # Apply tranformations, move to device and unsqueeze to insert a dimension of size one at position 0 eg: [3,224,224] -> [1,3,224,224]
    image = transform(img).unsqueeze(0)
    # Predict the image
    output = model_t_s(image)
    # Move the output to cpu, change to numpy array and get the index having max value
    index = output.data.numpy().argmax()
    # Get the label
    pred = birds[index]
    print(pred)
    response = jsonify({'prediction': pred})
    response.status_code = 201
    return response
    # OLD CODE TO SAVE IN UPLOADS
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    # pred = pre_processing(filename)


    # OLD CODE TO SAVE IN DB
    # converted_string = base64.b64encode(file.read())
    # connection = psycopg2.connect(user="qostkcboaonzuv",password="8e5c2b15ffab4b95056913b8756721c4c28266a104d04e0c188da1d3df6fec4a",host="ec2-44-194-117-205.compute-1.amazonaws.com",port="5432",database="dcferq0mj933p3") 
    # cursor = connection.cursor()
    # cursor.execute("INSERT INTO images(id,imgname, img) VALUES (DEFAULT,%s,%s) RETURNING id", (filename, converted_string))
    # returned_id = cursor.fetchone()[0]
    # connection.commit()
    # print("Stored {0} into DB record {1}".format(filename, returned_id))    
  else:
    response = jsonify({"message": "Allowed filetypes are "+ " ".join(ALLOWED_EXTENSIONS)})
    response.status_code = 400
    return response

if __name__ == "__main__":
  app.run()