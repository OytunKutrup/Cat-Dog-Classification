import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import tensorflow as tf

#Modelin yuklenmesi
model = tf.keras.models.load_model("2x64x0-CNN.model")

IMG_SIZE = 50
CATEGORIES = ["Kopek", "Kedi"]

#Gui nin olusturulmasi
top=tk.Tk()
top.geometry('800x600')
top.title('Yapay Zeka Yontemleri ve Uygulamalari Kedi-Kopek Siniflandirmasi')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',17,'bold'))
sign_image = Label(top)

#Konumu verilen verinin test icin hazirlanmasi
def prepare(file_path):
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#Verinin siniflandirilmasi

def classify(file_path):
    class_name = model.predict_classes([prepare(file_path)])
    result = CATEGORIES[int(class_name[0][0])]
    print(result)
    label.configure(foreground='#011638', text=result)

def show_classify_button(file_path):
    classify_b=Button(top,text="Fotografi Siniflandir",
   command=lambda: classify(file_path),
   padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Fotograf yukle",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Kedi-Kopek Siniflandirmasi",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

model.summary()