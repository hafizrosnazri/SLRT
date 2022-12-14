# Overview
Real-time Vision-based Sign Language to Text Interpreter by Using Artificial Intelligence and Augmented Reality Element by Muhammad Hafizuddin Rosnazri

Real-time Vision-based Sign Language to Text Interpreter by Using Artificial Intelligence and Augmented Reality Element is a project that can interpret sign language to text in real-time. This communicator used a machine learning approach with a slight touch of deep learning elements, which used OpenCV, MediaPipe, and Tensorflow algorithms. Those algorithms have been used to differentiate the hand from other objects, detect movement and produce Augmented Reality hand landmarks on the hand, and perform imagery data analysis to produce output in real-time. The camera will detect the user’s hand movement, and the output will be produced on the LCD monitor. This project has been developed using the Python programming language with Thonny Python IDE as the integrated development environment. 13,000 of ASL’s alphabets and 5,000 of ASL’s number imagery datasets have been trained by using a cloud platform, Google Colab. The training process for the alphabets produced 99.85% accuracy and 100% accuracy for the numbers. At the end of this project, the construction of a machine learning algorithm able to produce alphabets and numbers on LCD monitor in real-time and a workable prototype have been developed by using a USB camera, LCD monitor, and a Raspberry Pi microcontroller. The output in the form of text appeared on the LCD monitor by demonstrating ASL’s alphabet and number hand gestures. The performance of the prototype has been analyzed and experimented with by two users at plain and noise backgrounds with different determined distances. 

# Requirements
## Hardware
1) Computer / Laptop
2) Camera / USB Camera
3) Monitor
## Software
1) Python IDE
   - PyCharm
   - Thonny Python
   - MS Visul Studio
   - etc
2) Python Packages
   - TKinter
   - Pillow
   - OpenCV (cv2)
   - Tensorflow Keras
   - NumPy
   - Pandas
   - Mediapipe
3) Environment (OS)
   - Windows
   - Linux (If you are going to run the algorithms in Linux environment, make sure that you need to convert the algorithm files from **MS-DOS** to **Unix**)

# How to Run?
1) Download all of the packages https://github.com/hafizrosnazri/SLRT/tree/master/Packages and save it in your local machine
2) Before running the algorithm, make sure that all of the required Python packages have been isntalled
3) Open **sign_language_real_time_communicator.py** and execute the algorithm
4) Wait for the algorithm to be processed
5) A GUI to select interpretation will popped up
6) Choose either **Alphabet Interpretation** or **Number Interpretation**
7) A live video feed will popped up
8) Demonstrate the ASL alphabet or number hand gesture
9) Press **"q"** button on your keyboard to close the video feed and click **Stop** button on the GUI to stop the process

# ASL Reference
## ASL Alphabets
<p align="left">
  <img src="Packages/ASL ALPHABETS.jpg" width="350" title="hover text">
</p>

## ASL Numbers
<p align="left">
  <img src="Packages/ASL NUMBERS.png" width="350" title="hover text">
</p>

# Limitations
1) 1 hand detection and interpretation at a time
2) Detection distance up to 3 meters and interpretation distance up to 1.5 meters
3) Lighting, camera resolution, processing power, and hand features may affect the results

# Contributors
Muhammad Hafizuddin Rosnazri, Anis Nabilah Shahrul Yazid, Intel Malaysia Makers Club
