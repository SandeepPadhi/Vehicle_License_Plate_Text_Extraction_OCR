# Vehicle-License-Plate-Text-Extraction_OCR


'''
Install following libraries in your system:
cv2
numpy
pytesseract - pip install pytesseract
imutils
os


Abstract:
The Project is Vehicle License Number Plate's Text Extraction System using Image Processing and Optical Character 
(OCR).Image Processing is achieved using OpenCV.For OCR , Pytesseract (OCR engine supported by google) is used.


For Tutorial on OCR using Pytesseract go the following tutorial:
https://nanonets.com/blog/ocr-with-tesseract/

Follow the Link below to find out best practices for using pytesseract:
https://ai-facets.org/tesseract-ocr-best-practices/


Description of Project Folder:
Following is the Tree structure of the Project:
License_Plate_OCR
    - crop_h1
        -All crop* Folders contains images
    - crop_h2
    - crop_h3
    - crop_h4
    - crop_m1
    - crop_m2
    - crop_m3
    - crop_m4

    -trainVal.csv :Contains text on the images in different crop* Folders
    -Tesseract_files
    -Tesseract_html

    -OcrAnalyser.py
        -This is the main file which contains the code running the project.


Execution:
In the working directory containing OcrAnalyzer.py,open the terminal and type:
                    python OcrAnalyzer.py
        

NOTE:
    Windows containing image processing output will pop up.
    Press any key to move ahead with output.Keep pressing the any key.




Ouput:
1.It outputs text on the license plate along with image no on the command line.
2.Ouputs 4 imageprocessing windows showing 
    1.Grayscale image
    2.Thresholded image
    3.Image showing contours drawn on original image
    4.Binary Image containing text on license plate
    
    
    Output1.png and Output2.png in the project structure shows the output of the execution.




'''

