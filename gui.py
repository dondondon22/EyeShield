import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon  # For icons if needed
from PyQt5.QtCore import Qt  # For alignments

class EyeShieldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeShield - DR Screening System")
        self.setGeometry(100, 100, 800, 600)  # Optimized for standard monitors (Table 4)
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.login_page = LoginPage(self)
        self.main_menu_page = MainMenuPage(self)
        self.instructions_page = InstructionsPage(self)
        self.upload_page = UploadPage(self)
        self.results_page = ResultsPage(self)  # Initialized but updated dynamically

        self.central_widget.addWidget(self.login_page)
        self.central_widget.addWidget(self.main_menu_page)
        self.central_widget.addWidget(self.instructions_page)
        self.central_widget.addWidget(self.upload_page)
        self.central_widget.addWidget(self.results_page)

        self.central_widget.setCurrentWidget(self.login_page)  # Start with Landing (Figure 10)

    def show_page(self, page):
        self.central_widget.setCurrentWidget(page)

class LoginPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Login to EyeShield", alignment=Qt.AlignmentFlag.AlignCenter))
        self.username = QLineEdit(self)
        self.username.setPlaceholderText("Username")
        layout.addWidget(self.username)
        self.password = QLineEdit(self)
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password)
        login_btn = QPushButton("Login")
        login_btn.clicked.connect(self.login)
        layout.addWidget(login_btn)
        self.setLayout(layout)

    def login(self):
        # Mock auth (replace with secure logic, e.g., local SQLite for privacy)
        if self.username.text() == "clinician" and self.password.text() == "password":
            self.parent.show_page(self.parent.main_menu_page)
        else:
            QMessageBox.critical(self, "Error", "Invalid credentials")

class MainMenuPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Main Menu", alignment=Qt.AlignmentFlag.AlignCenter))
        instructions_btn = QPushButton("Instructions")
        instructions_btn.clicked.connect(lambda: self.parent.show_page(self.parent.instructions_page))
        layout.addWidget(instructions_btn)
        upload_btn = QPushButton("Upload Fundus Image")
        upload_btn.clicked.connect(lambda: self.parent.show_page(self.parent.upload_page))
        layout.addWidget(upload_btn)
        self.setLayout(layout)

class InstructionsPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Instructions Page", alignment=Qt.AlignmentFlag.AlignCenter))
        layout.addWidget(QLabel("1. Upload clear fundus images.\n2. System rejects ungradable ones.\n3. Review results with heatmaps."))
        back_btn = QPushButton("Back to Menu")
        back_btn.clicked.connect(lambda: self.parent.show_page(self.parent.main_menu_page))
        layout.addWidget(back_btn)
        self.setLayout(layout)

class UploadPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Upload Page", alignment=Qt.AlignmentFlag.AlignCenter))
        upload_btn = QPushButton("Select Image")
        upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(upload_btn)
        back_btn = QPushButton("Back to Menu")
        back_btn.clicked.connect(lambda: self.parent.show_page(self.parent.main_menu_page))
        layout.addWidget(back_btn)
        self.setLayout(layout)

    def upload_image(self):
        file_path = QFileDialog.getOpenFileName(self, "Select Fundus Image", "", "Images (*.jpg *.png)")[0]
        if file_path:
            # Integrate backend (Sprint 4): Call preprocess, inference from models.py
            # Mock result for now
            result_data = {"grade": "Moderate DR", "confidence": 0.95, "heatmap": "path/to/overlay.png"}
            if result_data["confidence"] < 0.8:  # Uncertainty rejection (Objective 2.2)
                QMessageBox.information(self, "Rejected", "Image ungradable - please retake.")
            else:
                self.parent.results_page.update_results(result_data)
                self.parent.show_page(self.parent.results_page)

class ResultsPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()
        self.results_label = QLabel("Screening Results", alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.results_label)
        self.details_label = QLabel()
        self.layout.addWidget(self.details_label)
        self.image_label = QLabel()  # For heatmap display
        self.layout.addWidget(self.image_label)
        back_btn = QPushButton("Back to Menu")
        back_btn.clicked.connect(lambda: self.parent.show_page(self.parent.main_menu_page))
        self.layout.addWidget(back_btn)
        self.setLayout(self.layout)

    def update_results(self, data):
        self.details_label.setText(f"DR Severity: {data['grade']}\nConfidence: {data['confidence']:.2f}")
        pixmap = QPixmap(data['heatmap']).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern theme for clinician trust
    window = EyeShieldApp()
    window.show()
    sys.exit(app.exec())