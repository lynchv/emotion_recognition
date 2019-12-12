import os
from shutil import copyfile

labelPath = "D:\cohn-kanade\Emotion"

for root, dirs, files in os.walk(labelPath):
    for file in files:
        if file.endswith(".txt"):
            filePath = os.path.join(root, file)
            newFilePath = (os.path.join(root, file)).replace("Emotion", "Images&Labels_SVM")
            copyfile(filePath, newFilePath)
