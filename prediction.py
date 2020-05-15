import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import keras
from keras.preprocessing import image



def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1])).pack()
    panel_image = tk.Label(frame, image=img).pack()
    

def classify():
    original = Image.open(image_data)
    original = original.resize((150, 150), Image.ANTIALIAS)
    model = keras.models.load_model('model\\model.h5')
    image_test = image.img_to_array(original)
    image_test = image_test.reshape((1,) + image_test.shape)
    image_test =image_test.astype('float32') / 255
    rust_prob = model.predict(image_test)
    #print(rust_prob)
    if (rust_prob > 0.50):
        pred = "This is a Rust image"
        # depth = 15
        # thresh_hold = 0.8
        # distance = 5
        # thresh = 0.07
        # img = Red_particles.scale_image(image_path,max_size=1000000)
        # Red_particles.energy_gLCM(img,depth,thresh_hold,distance,thresh)
    else:
        pred = "This is a no Rust image"
    label = pred
    table = tk.Label(frame, text="Result:").pack()
    result = tk.Label(frame,text= str(label).upper()).pack()
         
root = tk.Tk()
root.title('Corossion Detector')
#root.iconbitmap('class.ico')
root.resizable(False, False)

tit = tk.Label(root, text="Choose an image and press Classify Image", padx=25, pady=6, font=("", 12)).pack()

canvas = tk.Canvas(root, height=800, width=1000, bg='#000080')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.6, relheight=0.5, relx=0.2, rely=0.25)

chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)

class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)


root.mainloop()