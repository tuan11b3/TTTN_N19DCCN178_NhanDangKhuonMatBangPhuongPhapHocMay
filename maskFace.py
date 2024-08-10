import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from io import BytesIO
from skimage import io
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Tải mô hình
model = load_model('XceptionModel_224.h5')
# model = load_model('XceptionModel128.h5')
# IMG_SIZE = 128  # Kích thước hình ảnh đầu vào cho mô hình
IMG_SIZE = 224

# Tạo đối tượng nhận diện khuôn mặt
face_model = cv2.CascadeClassifier('C:/Users/ASUS/Documents/TTTN_D20_2024/maskDetection/Haarcascade/haarcascade_frontalface_default.xml')

def getFaces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8)
    return faces

def newSize(width, height):
    if width < 600:
        return newSize(width * 1.12, height * 1.12)
    if width >= 1200:
        return newSize(width / 1.12, height / 1.12)
    return int(width), int(height)

def AdjustSize(f):
    img = Image.open(f)
    width, height = img.size
    new_width, new_height = newSize(width, height)
    return new_width, new_height

def Draw(img, face):
    (x, y, w, h) = face
    mask_label = {0: 'Has Mask!', 1: 'No Mask'}
    label_color = {0: (0, 255, 0), 1: (255, 0, 0)}

    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return img

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = np.expand_dims(crop, axis=0) / 255.0
    mask_result = model.predict(crop)
    pred_label = 0 if mask_result[0][0] < 0.5 else 1
    confidence = mask_result[0][0] if pred_label == 1 else 1 - mask_result[0][0]

    cv2.putText(img, f"{mask_label[pred_label]} ({confidence*100:.2f}%)",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color[pred_label], 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), label_color[pred_label], 2)
    return img

def MaskDetection(imgUri):
    response = requests.get(imgUri)
    f = BytesIO(response.content)
    img = io.imread(f)
    resize = AdjustSize(f)
    img = cv2.resize(img, resize)
    faces = getFaces(img)

    if len(faces) >= 1:
        for face in faces:
            Draw(img, face)
        plt.figure(figsize=(16, 14))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("No Face!")

def MaskDetectionFromFile(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print("Failed to load image!")
        return
    resize = AdjustSize(file_path)
    img = cv2.resize(img, resize)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = getFaces(img_rgb)

    if len(faces) >= 1:
        for face in faces:
            img_rgb = Draw(img_rgb, face)
        plt.figure(figsize=(16, 14))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
    else:
        print("No Face!")

def detect_from_camera():
    cap = cv2.VideoCapture(0)  # Mở camera mặc định
    window_name = 'Face Mask Detection'

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.addWeighted(frame, 1.5, frame, 0, 0)

        faces = getFaces(frame)
        if len(faces) >= 1:
            for face in faces:
                (x, y, w, h) = face
                face_crop = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                crop = np.reshape(face_rgb, [1, IMG_SIZE, IMG_SIZE, 3]) / 255.0

                mask_result = model.predict(crop)
                pred_label = 0 if mask_result[0][0] < 0.5 else 1
                confidence = mask_result[0][0] if pred_label == 1 else 1 - mask_result[0][0]

                mask_label = {0: 'Has Mask!', 1: 'No Mask'}
                label_color = {0: (0, 255, 0), 1: (255, 0, 0)}
                cv2.putText(frame, f"{mask_label[pred_label]} ({confidence * 100:.2f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color[pred_label], 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), label_color[pred_label], 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detection")

        # Khởi tạo chế độ giao diện
        self.theme = "light"

        # Tạo các thuộc tính giao diện
        self.create_widgets()
        self.update_theme()

    def create_widgets(self):
        # Khung tiêu đề
        self.header_frame = tk.Frame(self.root, height=80)
        self.header_frame.pack(fill=tk.X, padx=20, pady=10)

        # Thêm biểu tượng và tiêu đề
        self.icon_image = tk.PhotoImage(file="icons/PTITHCM_icon.png")  # Đặt đường dẫn đến biểu tượng của bạn
        self.icon_label = tk.Label(self.header_frame, image=self.icon_image)
        self.icon_label.pack(side=tk.LEFT, padx=(0, 10))

        # self.title_label = tk.Label(self.header_frame, text="Face Mask Detection\nSV: Nguyễn Anh Tuấn \nMSSV: n19dccn178", font=("Helvetica", 16), justify=tk.LEFT)
        self.title_label = tk.Label(self.header_frame,text="Face Mask Detection", font=("Helvetica", 20, "bold"))
        self.title_label.pack(side=tk.LEFT,anchor="w")

        self.info_label = tk.Label(self.header_frame, text="SV: Nguyễn Anh Tuấn\nMSSV: n19dccn178",
                                   font=("Helvetica", 14), bg="lightgray", justify=tk.LEFT)
        self.info_label.pack(side=tk.LEFT, anchor="w")

        # Khung chứa các nút với kích thước cố định
        self.button_frame = tk.Frame(self.root, width=400, height=300)
        self.button_frame.pack_propagate(False)  # Ngăn không cho khung tự động điều chỉnh kích thước theo các widget con
        self.button_frame.pack(fill=tk.BOTH, padx=20, pady=10)

        # Bố trí các nút với grid
        button_labels = ["Detect from URL", "Detect from File(s)", "Detect from Camera", "Switch Theme"]
        self.buttons = []

        for i, label in enumerate(button_labels):
            button = tk.Button(self.button_frame, text=label, command=self.create_command(label), width=20, height=2)
            button.grid(row=i // 2, column=i % 2, sticky="nsew", padx=10, pady=10)
            self.buttons.append(button)

        # Tạo tỷ lệ đều cho các hàng và cột
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.rowconfigure(0, weight=1)
        self.button_frame.rowconfigure(1, weight=1)
        self.button_frame.rowconfigure(2, weight=1)
        self.button_frame.rowconfigure(3, weight=1)

    def update_theme(self):
        if self.theme == "light":
            self.background_color = "white"
            self.label_color = "blue"
            self.text_color = "black"
            self.button_bg = "lightgray"
            self.button_fg = "black"
        else:
            self.background_color = "black"
            self.label_color = "yellow"
            self.text_color = "white"
            self.button_bg = "gray"
            self.button_fg = "white"

        # Cập nhật màu sắc cho các thành phần
        self.root.config(bg=self.background_color)
        self.header_frame.config(bg=self.background_color)
        self.title_label.config(bg=self.background_color, fg=self.label_color)
        self.button_frame.config(bg=self.background_color)

        for button in self.buttons:
            button.config(bg=self.button_bg, fg=self.button_fg)

    def create_command(self, label):
        commands = {
            "Detect from URL": self.detect_from_url,
            "Detect from File(s)": self.detect_from_file,
            "Detect from Camera": self.detect_from_camera,
            "Switch Theme": self.switch_theme
        }
        return commands.get(label)

    def detect_from_url(self):
        url = simpledialog.askstring("Input", "Enter the image URL:")
        if url:
            MaskDetection(url)

    def detect_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            MaskDetectionFromFile(file_path)

    def detect_from_camera(self):
        detect_from_camera()

    def switch_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.update_theme()

# Khởi chạy ứng dụng
root = tk.Tk()
app = App(root)
root.mainloop()
