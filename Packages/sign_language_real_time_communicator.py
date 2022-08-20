from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import cv2
import imutils
import mediapipe as mp
from generate_csv import get_connections_list, get_distance
from tensorflow import keras
import numpy as np
import pandas as pd
import time
import pickle


def Alphabet_Interpretation():
    
    if selected.get() == 1:
        def get_sign_list():
            
            # Function to get all the values in SIGN column
            df = pd.read_csv('alphabets.csv', index_col=0)
            return df['SIGN'].unique()

        def real_time_prediction():
            sign_list = get_sign_list()
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            connections_dict = get_connections_list()
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Initialize the fps
            pTime = 0
            cTime = 0
            
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8) as hands:
                while cap.isOpened():
                    
                    # Get image from webcam, change color channels and flip
                    ret, frame = cap.read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.flip(image, 1)
                    
                    # Get result
                    results = hands.process(image)
                    if not results.multi_hand_landmarks:

                        # If no hand detected, then just display the webcam frame
                        cv2.imshow('Alphabets Interpretation', frame)

                    else:
                        
                        # If hand detected, superimpose landmarks and default connections
                        mp_drawing.draw_landmarks(
                            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                        )
                        
                        cTime = time.time()
                        fps = 1/(cTime - pTime)
                        pTime = cTime

                        # Get landmark coordinates & calculate length of connections
                        coordinates = results.multi_hand_landmarks[0].landmark
                        data = []
                        for _, values in connections_dict.items():
                            data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                        
                        # Scale data
                        data = np.array([data])
                        data[0] /= data[0].max()
                        
                        # Load model from h5 file
                        model = keras.models.load_model('alphabets_ann_model.h5')

                        # Get prediction
                        pred = np.array(model(data))
                        pred = sign_list[pred.argmax()]
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                        # Display text showing prediction with fps
                        image = cv2.rectangle(image, (200, 415), (500, 460), (0, 0, 0), cv2.FILLED)
                        image = cv2.putText(image, f'Alphabet {str(pred)}', (210, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                        image = cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                        # Display final image
                        cv2.imshow('Alphabets Interpretation', image)

                    # Press Q on keyboard to quit
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()

        if __name__ == "__main__":
            real_time_prediction()

        
        btnEnd.configure(state = "active")
        rad1.configure(state = "disabled")
        rad2.configure(state = "disabled")
        
def Number_Interpretation():
    
    if selected.get() == 2:
        
        def get_sign_list():
            
            # Function to get all the values in SIGN column
            df = pd.read_csv('numbers.csv', index_col=0)
            return df['SIGN'].unique()

        def real_time_prediction():
            sign_list = get_sign_list()
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            connections_dict = get_connections_list()

            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Initialize the fps
            pTime = 0
            cTime = 0
            
            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8) as hands:
                while cap.isOpened():
                    # Get image from webcam, change color channels and flip
                    ret, frame = cap.read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.flip(image, 1)

                    # Get result
                    results = hands.process(image)
                    if not results.multi_hand_landmarks:
                        
                        # If no hand detected, then just display the webcam frame
                        cv2.imshow('Numbers Interpretation',frame)
                        
                    else:
                        
                        # If hand detected, superimpose landmarks and default connections
                        mp_drawing.draw_landmarks(
                            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                        )
                        
                        cTime = time.time()
                        fps = 1/(cTime - pTime)
                        pTime = cTime
                        
                        # Get landmark coordinates & calculate length of connections
                        coordinates = results.multi_hand_landmarks[0].landmark
                        data = []
                        for _, values in connections_dict.items():
                            data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                        
                        # Scale data
                        data = np.array([data])
                        data[0] /= data[0].max()
                        
                        # Load model from h5 file
                        model = keras.models.load_model('numbers_ann_model.h5')

                        # Get prediction
                        pred = np.array(model(data))
                        pred = sign_list[pred.argmax()]
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                        # Display text showing prediction with fps
                        image = cv2.rectangle(image, (200, 415), (455, 460), (0, 0, 0), cv2.FILLED)
                        image = cv2.putText(image, pred, (210, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                        image = cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                        # Display final image
                        cv2.imshow('Numbers Interpretation', image)

                    # Press Q on keyboard to quit
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()


        if __name__ == "__main__":
            real_time_prediction()

        
        btnEnd.configure(state = "active")
        rad1.configure(state = "disabled")
        rad2.configure(state = "disabled")
  
def stop_button():
    lblVideo.image = ""
    lblInfoVideoPath.configure(text = "")
    rad1.configure(state = "active")
    rad2.configure(state = "active")
    selected.set(0)
    cap.release()

# Tkinter
cap = None
root = Tk()
root.title('SLRT')
root.iconbitmap('icon.ico')

lblInfo1 = Label(root, text = "SIGN LANGUAGE REAL-TIME COMMUNICATOR", font = "bold")
lblInfo1.grid(column = 0, row = 0, columnspan = 2)

selected = IntVar()
rad1 = Radiobutton(root, text = "Alphabet Interpretation", width = 20, value = 1, variable = selected, command = Alphabet_Interpretation)
rad1.grid(column = 0, row = 1)
rad2 = Radiobutton(root, text = "Number Interpretation", width = 20, value = 2, variable = selected, command = Number_Interpretation)
rad2.grid(column = 1, row = 1)

lblInfoVideoPath = Label(root, text = "", width = 20)
lblInfoVideoPath.grid(column = 0, row = 2)

lblVideo = Label(root)
lblVideo.grid(column = 0, row = 3, columnspan = 2)

btnEnd = Button(root, text = "Stop", state = "disabled", command = stop_button)
btnEnd.grid(column = 0, row = 4, columnspan = 2)

lblInfo2 = Label(root, text = "Developed by")
lblInfo2.grid(column = 0, row = 7, columnspan = 2)

lblInfo3 = Label(root, text = "Muhammad Hafizuddin Rosnazri")
lblInfo3.grid(column = 0, row = 8, columnspan = 2)

lblInfo4 = Label(root, text = "Anis Nabilah Shahrul Yazid")
lblInfo4.grid(column = 0, row = 9, columnspan = 2)

lblInfo5 = Label(root, text = " ")
lblInfo5.grid(column = 0, row = 10, columnspan = 2)

lblInfo5 = Label(root, text = "Developed for")
lblInfo5.grid(column = 0, row = 11, columnspan = 2)

lblInfo5 = Label(root, text = "Intel Malaysia IM50 GEEKCON 2022")
lblInfo5.grid(column = 0, row = 12, columnspan = 2)

root.mainloop()
