import cv2  # Importing the OpenCV library for computer vision tasks
import numpy as np  # Importing the NumPy library for numerical operations
import face_recognition  # Importing the face_recognition library for face recognition functionality
import os  # Importing the os module for interacting with the operating system
from datetime import datetime  # Importing the datetime module for timestamping

path = 'ImagesAttendance'  # Path to the directory containing attendance images
images = []  # List to store the loaded images
classNames = []  # List to store the class names (image filenames without extension)
myList = os.listdir(path)  # Getting the list of files in the attendance images directory
print(myList)  # Printing the list of attendance image files

# Loading the attendance images and storing them in the 'images' list
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)  # Printing the list of class names (attendance image filenames without extension)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting the image to RGB format for face recognition processing
        encode = face_recognition.face_encodings(img)[0]  # Encoding the facial features of the image
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(
    0)  # Capturing video from webcam (change the parameter to the video file path for processing a video file)

while True:
    success, img = cap.read()  # Reading frames from the video stream
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resizing the frame for faster face recognition processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Converting the frame to RGB format for face recognition processing

    facesCurFrame = face_recognition.face_locations(imgS)  # Detecting face locations in the current frame
    encodesCurFrame = face_recognition.face_encodings(imgS,
                                                      facesCurFrame)  # Encoding the facial features of the faces in the current frame

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,
                                                 encodeFace)  # Comparing the facial encodings with the known encodings
        faceDis = face_recognition.face_distance(encodeListKnown,
                                                 encodeFace)  # Calculating the face distance between the encodings
        matchIndex = np.argmin(faceDis)  # Finding the index with the minimum face distance

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # Getting the corresponding name for the matched face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing a rectangle around the detected face on
            # the original image
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Drawing a filled rectangle for
            # displaying the name label
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                        2)  # Adding the name
            # label to the image
            markAttendance(name)  # Marking the attendance for the recognized face

    cv2.imshow('Webcam', img)  # Displaying the processed frame with face recognition

    cv2.waitKey(1)  # Waiting for a key press (1ms delay) to exit the loop
