import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap  # For future image display (e.g., heatmaps)
from PyQt5.QtCore import Qt

class EyeShieldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeShield - DR Screening Starter")
        self.setGeometry(100, 100, 600, 400)  # Simple size for beginners
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        # Create pages
        self.login_page = LoginPage(self)
        self.main_menu_page = MainMenuPage(self)

        # Add to stack
        self.central_widget.addWidget(self.login_page)
        self.central_widget.addWidget(self.main_menu_page)

        self.central_widget.setCurrentWidget(self.login_page)  # Start with login

    def show_page(self, page):
        self.central_widget.setCurrentWidget(page)

class LoginPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        label = QLabel("Login to EyeShield", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.username = QLineEdit(self)
        self.username.setPlaceholderText("Username")
        layout.addWidget(self.username)
        self.password = QLineEdit(self)
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password)
        login_btn = QPushButton("Login", self)
        login_btn.clicked.connect(self.login)
        layout.addWidget(login_btn)
        self.setLayout(layout)

    def login(self):
        # Simple mock check (expand later for real auth)
        if self.username.text() == "clinician" and self.password.text() == "password":
            self.parent.show_page(self.parent.main_menu_page)
        else:
            QMessageBox.critical(self, "Error", "Invalid credentials")

class MainMenuPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        label = QLabel("Main Menu - Upload Image", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        upload_btn = QPushButton("Select Fundus Image", self)
        upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(upload_btn)
        self.setLayout(layout)

    def upload_image(self):
        file_path = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")[0]
        if file_path:
            # Placeholder for Sprint 2 integration (preprocess and quality check)
            # For now, just show a simple message
            QMessageBox.information(self, "Upload Success", f"Image selected: {file_path}\n(Next: Add quality rejection here)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeShieldApp()
    window.show()
    sys.exit(app.exec_())