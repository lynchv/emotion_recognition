import os
import imghdr
import cv2

# cascadePath = 'C:\\Users\\vince\\Desktop\\sys843\\Source\\haarcascade_frontalface_default.xml'
# dataSetPath = "C:\\Users\\vince\\Desktop\\cohn-kanade\\images"
cascadePath = 'C:\\Users\\Vincent\\Desktop\\sys843\\haarcascade_frontalface_default.xml'
dataSetPath = 'D:\\cohn-kanade\\Images_SVM'


def deal_with_image(imgpath):
    gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(gray, 1.25, 5)
    if len(faces) == 0:
            print(imgpath)  # Change parameters of detectMultiScale or manually crop the image
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(imgpath, roi_gray)


def trav_dir(dirpath):
    os.chdir(dirpath)
    dir_list = os.listdir()

    # travers current directory and if directoy found call itself
    for x in dir_list:
        if(os.path.isdir(x)):
            trav_dir(x)
        # imghdr.what return mime type of the image
        elif(imghdr.what(x) in ['png']):
            deal_with_image(x)

    # reached directory with no directory
    os.chdir('./..')


face_cascade = cv2.CascadeClassifier(cascadePath)
trav_dir(dataSetPath)
