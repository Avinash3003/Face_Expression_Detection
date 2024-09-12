import telebot
from PIL import Image
import numpy as np
import pickle
import io

# model=pickle.load(open('model.pkl','rb')) 
le=pickle.load(open('label_encoder1.pkl','rb'))

# Load model
from keras.models import load_model
model= load_model('model1.h5')



botToken='6961394073:AAGHH5jqi04wqoLkgu203BPlWczBe0MPvHw'
bot=telebot.TeleBot(botToken,parse_mode=None)


@bot.message_handler(commands=['start'])
def send_welcome(message):   
    bot.reply_to(message,"Welcome to my bot\nGive me your image as input\nI can guess what is your facial_expression😉")

@bot.message_handler(content_types=['photo'])
def echo_image(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        downloaded_file = bot.download_file(file_path)
        img_bytes = io.BytesIO(downloaded_file)


        img = Image.open(img_bytes)
        img = img.resize((48, 48))
        
        img = img.convert('L')
        img = np.array(img)
        print(img,img.shape)
        img=np.array(img).reshape(1,48,48,1)
        img=img/255.0
        print(img,img.shape)

        pred = model.predict(img)
        predicted_label = le.inverse_transform([pred.argmax()])[0]
        print(predicted_label)

        # bot.reply_to(message, "Your photo")
        bot.reply_to(message,"Your Expression is "+predicted_label)
        # bot.send_photo(message.chat.id, downloaded_file)
    except Exception as e:
        bot.reply_to(message, "An error occurred. Please try again.")


bot.polling()









