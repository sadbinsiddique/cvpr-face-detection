
## Attendance Using Face Detection

An automated class attendance system based on **Computer Vision and Pattern Recognition**. This project detects and recognizes faces in real-time to automatically mark attendance, making manual roll calls obsolete.

## 📌 Features

- 🎯 Real-time face detection using YOLOvX
- 📅 Automatic attendance logging with timestamp
- 🧑‍🎓 Student face registration and dataset management
- 📝 Attendance report generation (CSV or database)
- 🖥️ GUI-based or command-line interface (optional)

## 🧠 Model

This system integrates:
- **YOLO** for high-speed and accurate face detection
- **OpenCV** for image preprocessing and video handling
- **Custom dataset** for face recognition

## 🗂️ Dataset

- All training/testing images are stored in the `dataset/` folder
- Folder structure can be:
  - Flat (all images in one folder)
  - Subfolders by student name (recommended for recognition)
- Supported formats: `.jpg`, `.png`

## 📂 Project Structure

```
cvpr-face-detection/
│
├── dataset/              # Dataset containing student face      images
├── attendance/           # Saved attendance records
├── weights/              # trained model weights
├── main.py               # Application script for detection and attendance
├── register.py           # Script to add new students
├── train.py              # YOLOvX training script 
├── detect.py             # YOLOvX detection script
├── utils.py              # Helper functions (detection, logging)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip (Python package installer)
- OpenCV
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sadbinsiddique/cvpr-face-detection.git

   cd FaceAttend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


### Run Detection & Attendance

```bash
python detect.py --source webcam --weights weights/best.pt
```

## 🧪 Sample Workflow

1. Add student images to the `dataset/` folder.
2. Train or load a model.
3. Run `detect.py` to start face detection and attendance marking.
4. Logs are stored with date and time in the `attendance/` folder.

## 📊 Output

- `.csv` or database files logging attendance
- Model weights stored in `weights/`

## 🔒 Disclaimer

This project is for educational and experimental purposes only. Ensure you comply with privacy laws and obtain consent before using facial recognition in real scenarios.

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

For questions, suggestions, or feedback:  
🔗 GitHub: [sadbinsiddique](https://github.com/sadbinsiddique)   [mdjnaim](https://github.com/mdjnaim)
