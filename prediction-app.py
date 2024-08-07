from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QSplashScreen, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QSizePolicy
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pydicom
import sys

class PneumoniaDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pneumonia Detector")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #ebebeb;")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        radio_layout = QHBoxLayout()
        self.radio_model1 = QRadioButton("Only Pneumonia")
        self.radio_model2 = QRadioButton("Multiple Diseases")
        self.radio_model1.setStyleSheet("color: black")
        self.radio_model2.setStyleSheet("color: black")
        self.radio_model1.setChecked(True)
        radio_layout.addWidget(self.radio_model1)
        radio_layout.addWidget(self.radio_model2)

        self.radio_model1.toggled.connect(self.load_model)
        self.radio_model2.toggled.connect(self.load_model)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.image_label)

        logo = "/Users/yunus/Downloads/WhatsApp Image 2024-07-25 at 12.54.45.jpeg"
        pixmap = QPixmap(logo)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        main_layout.addLayout(radio_layout)

        button_layout = QHBoxLayout()
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.setStyleSheet("background-color: #b00909; color: white;")
        self.load_image_button.clicked.connect(self.load_image)
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setStyleSheet("background-color: #b00909; color: white;")
        self.predict_button.clicked.connect(self.predict_image)
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.predict_button)
        main_layout.addLayout(button_layout)

        self.model = None
        self.image_path = None

        self.load_model()

    def load_model(self):
        if self.radio_model1.isChecked():
            model_path = "/Users/yunus/Desktop/Pneumonia Detection App/Models/only-pneumonia-model.keras"
        elif self.radio_model2.isChecked():
            model_path = "/Users/yunus/Desktop/Pneumonia Detection App/Models/multiple-diseases-model.keras"
        
        if model_path:
            try:
                self.model = load_model(model_path)
                model_name = "Only Pneumonia" if self.radio_model1.isChecked() else "Multiple Diseases"
                self.setWindowTitle(f"Pneumonia Detector - {model_name} Loaded")
            except ValueError as e:
                self.result_label.setText(f"Error loading model: {str(e)}")
                self.result_label.setStyleSheet("color: red")


    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png *.jpeg *.dcm)")
        if fname:
            self.image_path = fname
            pixmap = QPixmap(fname)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def preprocess_image(self, image_path, target_size=(224, 224)):
        image = None
        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            image = cv2.imread(image_path)
        elif image_path.lower().endswith('dcm'):
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array
        if image is not None:
            image = cv2.resize(image, target_size)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
        return image

    def predict_image(self):
        if self.model and self.image_path:
            image = self.preprocess_image(self.image_path)
            if image is not None:
                prediction = self.model.predict(image)
                predicted_class = np.argmax(prediction, axis=1)[0]
                if self.radio_model1.toggled():
                    class_names = ['NORMAL', 'PNEUMONIA']
                else:
                    class_names = []
                result = class_names[predicted_class]
                self.result_label.setText(predicted_class)
            else:
                self.result_label.setText("Error processing image.")
        else:
            self.result_label.setText("Model or image not loaded.")
        self.result_label.setStyleSheet("color: black")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("/Users/yunus/Downloads/WhatsApp Image 2024-07-25 at 12.54.45.jpeg"))
    window = PneumoniaDetector()
    window.show()
    sys.exit(app.exec_())
