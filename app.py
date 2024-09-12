from flask import Flask,render_template,request,url_for
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

app = Flask(__name__)


# cnn=pickle.load(open('model.pkl','rb'))
from keras.models import load_model
cnn= load_model('model1.h5')

le=pickle.load(open('label_encoder1.pkl','rb'))


@app.route('/')
def Home():
    return render_template('home.html')


@app.route('/home',methods=['POST'])
def home():
    file = request.files['file']
    img = Image.open(file)
    img = img.resize((48, 48))
    img = img.convert('L')
    img = np.array(img)
    print(img,img.shape)
    img=np.array(img).reshape(1,48,48,1)
    img=img/255.0
    print(img,img.shape)

    pred = cnn.predict(img)
    predicted_label = le.inverse_transform([pred.argmax()])[0]
    print(predicted_label)
    return "The predicted value : "+predicted_label


if __name__=='__main__':
    app.run(debug=True)