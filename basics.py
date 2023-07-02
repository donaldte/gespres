import cv2  # Importing the OpenCV library for computer vision tasks
import face_recognition  # Importing the face_recognition library for face recognition functionality

# Loading and converting the image of user to RGB format for face recognition processing
imgUser = face_recognition.load_image_file('ImagesBasic/donald1.jpg')
imgUser = cv2.cvtColor(imgUser, cv2.COLOR_BGR2RGB)

# Loading and converting the image of user for to testing to RGB format for face recognition processing
imgTest = face_recognition.load_image_file('ImagesBasic/donald2.png')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detecting the face location and encoding the facial features of Elon Musk
faceLoc = face_recognition.face_locations(imgUser)[0]
encodeElon = face_recognition.face_encodings(imgUser)[0]
cv2.rectangle(imgUser, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # Drawing a rectangle around the detected face on the image

# Detecting the face location and encoding the facial features of Bill Gates
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)  # Drawing a rectangle around the detected face on the image

# Comparing the facial encodings of Elon Musk and Bill Gates to determine if they match
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

print(results, faceDis)  # Printing the comparison results and the face distance

# Adding text to the image of Bill Gates to display the comparison results and face distance
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Displaying the image of Elon Musk with the detected face rectangle
cv2.imshow('Elon Musk', imgUser)

# Displaying the image of Bill Gates with the detected face rectangle and comparison results
cv2.imshow('Elon Test', imgTest)

cv2.waitKey(0)  # Waiting for a key press to close the displayed images
