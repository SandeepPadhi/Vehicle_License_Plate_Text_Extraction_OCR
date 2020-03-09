import cv2
import numpy as np
import pytesseract
import imutils
import os 
import pandas as pd
df = pd.read_csv("trainVal.csv")
df= pd.read_csv("trainValOriginal.csv") 
df = df[['track_id','image_path','lp','train','Output']]# displying  dataframe - Output 1 


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




'''


#Converts image into Grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Converts Grayscale image to Binary using thresholding.It takes threshholding value and image as parameter
def thresholding(image,threshval):
    return cv2.threshold(image,threshval,255,cv2.THRESH_BINARY)[1]

#Does Morphological Function
def opening(image):
    kernel = np.ones((1,1),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#Does Dilation of Image
def dilate(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.dilate(image, kernel, iterations = 2)    

#Does Erosion of Image
def erode(image):
    kernel = np.ones((1,1),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#Performs Morphological Operation on the image.(The Below function is not used and is just kept for further improvement)
def morph(image):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return mask

#Converts list to String
def listToString(first):  
    str1 = "" 
    first=str1.join(first)
    return first

#OCR_Analyser Function AFter extracting text , which is mostly of two word(Firstword and secondword as name suggests)
#Firstwordpostprocessing does processing of firstword to remove noisy data as much as possible
#Secondwordpostprocessing does processing of secondword to remove noisy data as much as possible

def firstwordpostprocessing(value):
    #PostProcessing for firstword
    
    if len(value)==0:
        return ""
    #print("Before checking firstword:",value)

    firstword=value[0].split()
    firstword=firstword[0]
    firstword=[i for i in firstword]

    #print("After checking firstword:",firstword,",length:",len(firstword))
    if len(firstword)==1 or len(firstword)==0:
        return ""

    for key,val in enumerate(firstword):
        if val=='.' and firstword[key-1]=='1':
            firstword[key-1]='4'
            firstword.pop(key)
        elif val=='.':
            firstword.pop(key)
    
    if len(value)==1:
        firstword=listToString(firstword)
        return firstword[0]
    elif len(firstword)>=4 :
        firstword=firstword[len(firstword)-3:len(firstword)]


    if len(firstword)==0 or len(firstword)==1:
        return firstword

    #print("firstword:",listToString(firstword[0]))    
    #print("Len of firstword:",len(firstword))

    if firstword[1]=='8' or firstword[1]=='5' or firstword[1]=='3':
        firstword[1]='B'
    if firstword[1]=='2':
        firstword[1]='Z'
    if firstword[1]=='4':
        firstword[1]='L'
    firstword=listToString(firstword)
    return firstword

#Processing of the secondword
def secondwordpostprocessing(value):
    #Postprocessing for secondword
    secondword=""
    for i in range(1,len(value)):
        if value[i]!="" or value[i]!=" " or value[i]!=".":
            secondword=value[i].split()
            secondword=secondword[0]



    secondword=[i for i in secondword]
    if len(secondword)==0:
        return ""
    #print("Before checking secondword:",list(secondword))

    for key,val in enumerate(secondword):
        if val=='.' and secondword[key-1]=='1':
            secondword[key-1]='4'
            secondword.pop(key)
        elif val=='.':
            secondword.pop(key)

    #secondword=secondword[0:-1]
    
    for key,val in enumerate(secondword):
        if val=='?' or val=='Z':
            secondword[key]='2'
        if val=='/' or val=='A':
            secondword[key]='7'
        if val=='B':
            secondword[key]='8'
        if val=='L':
            secondword[key]='4'
        if val=='!':
            secondword[key]=='1'
        if val==';' and secondword[key-1]=='1':
            secondword[key-1]=4
            secondword.pop(key)
    secondword=listToString(secondword)
    #print("returning:",secondword)
    return secondword

#Returns path to an image
def getpath(Image_Folder,imageno):
    if imageno<=9:
        imageno='I0000'+str(imageno)
    elif imageno<=99:
        imageno='I000'+str(imageno)
    else:
        imageno='I00'+str(imageno)
    
    path=str(os.getcwd())+'/'+str(Image_Folder)+'/'+str(imageno)+'.png'
    #path='/home/sandeep/Downloads/IDFY/'+Image_Folder+'/'+imageno+'.png'
    return path

#Used to find avg value of grayscale image.This value will be used for thresholding the image to Binary.
def findavg(image):
    result=np.array(image).flatten()

    result=list(result)
    #print("Avg value normal:",sum(result)/len(result))
    avgthresh=sum(result)/len(result)
    indexval=0
    count=0

    for i in result:
        if i>avgthresh:
            indexval=count
            break
        count=count+1

    #avgthresh=result[indexval]
    #######Square mean root avg

    result=np.square(result)
    avgthreshsqrt=sum(result)/len(result)
    avgthreshsqrt=np.sqrt(avgthreshsqrt)
    
    #print("Avg square value sqrt:",avgthreshsqrt)
    avgthresh=avgthresh+avgthreshsqrt
    #avgthresh=avgthresh
    return avgthresh



#OCR_Analyser initiates,executes and produces the result.
#It takes image path and image no as input.
def OCR_Analyser(path,imageno):
    imageno=""
    image = cv2.imread(path)
    
    #Resigning the image to below dim as it has been found best during training.
    dim=(348,129)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    imageoriginal=image
    image=get_grayscale(image)
    avgthresh=findavg(image)

   #1
    #cv2.imshow('1-GrayScale:'+str(imageno),image)
    #cv2.waitKey(0)

    imagethresh=thresholding(image,avgthresh)
    imagethresh=opening(imagethresh)
    #2
    #cv2.imshow('2-Threshold:'+str(imageno),imagethresh)
    #cv2.waitKey(0)
    c=image
    contours, hierarchy = cv2.findContours(imagethresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    
    val=-1
    count=0
    for i in contours:
        if len(i)==4:
            val=count
            break
        count=count+1
    if len(contours)==0:
        #print("###############################")
        return "",imagethresh

    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(imageoriginal, c, -1, (0,255,0), 3)
    #3
    #cv2.imshow('3-Contours:'+str(imageno),imageoriginal)
    #cv2.waitKey(0)
    


    rect = cv2.minAreaRect(c)

  #Finds angle and rotates the image so that its horizontal.
    angle=rect[-1]
    if abs(angle)<90-abs(angle):
        angle=angle
    else:
        if angle<0:
            angle = 90-abs(angle)
        else:
            angle=abs(angle)-90

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imageproc= imutils.rotate(imagethresh,angle)
    contours, hierarchy = cv2.findContours(imageproc.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    cv2.drawContours(imageoriginal, c, -1, (0,255,0), 3)
    imageproc=imageproc[y:y+h,x:x+w]
    imageproc=erode(imageproc)
    #imageproc = cv2.medianBlur(imageproc,3)
    #imageproc=erode(imageproc)
    #imageproc=dilate(imageproc)


    #After Analysis I found pytesseract works best with dim= (385,90)
    imageproc = cv2.resize(imageproc, (385,90), interpolation = cv2.INTER_AREA) 
    #print("previous:",imageproc)

    for i in range(11):
        for j in range(384):
            imageproc[i][j]=255
            imageproc[89-i][j]=255
    for i in range(22):
        for k in range(89):
            imageproc[k][i]=255
            imageproc[k][384-i]=255


    #Pytesseract configuration 
    

    oem=0 #Legacy system
    psm=7 #Treats image as single text line
    custom_config = r'--oem '+str(oem)+' --psm '+str(psm)+'-c tessedit_char_whitelist=012345679abcdefghijklmnopqrstuvwlyz tessedit_char_blacklist=.-!/'
    value = pytesseract.image_to_string(imageproc, lang='eng', config=custom_config)
    value=value.encode('ascii','ignore')
    #4
    #cv2.imshow('4 - OCRImage:'+str(imageno),imageproc)
    #cv2.waitKey(0)
    value=value.split()
    finalresult=[]
    
    if len(value)>2:
        strval=" "
        for i in value:
            strval=strval+str(i)+" "
        finalresult=strval
    else:
        
        firstword=firstwordpostprocessing(value)
        secondword=""
        if len(value)>0 and len(value)>1:
            secondword=secondwordpostprocessing(value)

        #print("Firstword:",firstword)
        #print("secondword:",secondword)
        #print("Type of firstword:",type(str(firstword)))
        #print("Type of secondword:",type(secondword))
        
        finalresult=str(firstword)+" "+str(secondword)
    
    return finalresult,imageproc







#Enter the imageno and ImageFolder in below variables
imageno=0
Test_image_Folder='crop_m1'

Image_Folder=Test_image_Folder

#getpath() is used to get path to the image
path=getpath(Image_Folder,imageno)
print("Path :",path)
print("Pathnew:",os.getcwd())

#OCR_Analyser() analyser the result and extracts text from the image.
result,imageproc=OCR_Analyser(path,imageno)
print("The text on template is ",result)
pathnew=os.getcwd()

pathnew=pathnew+'/'+Image_Folder
no_of_images=len(os.listdir(pathnew))
print("Number of images in directory is ",no_of_images)

'''
#Loop below will through all the images in folder and extract text
for imageno in range(0,no_of_images):
    #print("IMAGE NUMBER:",imageno)
    #path=str(df['imagepath'])
    #path=getpath(Image_Folder,imageno)
    result,imageproc=OCR_Analyser(path,imageno)
    #print("The text on template is ",result)
    #df.set_value(imageno, "train", result)
    df['Output'][imageno]=result

    #print()
    #print()
    #cv2.imshow('finalimage2',imageproc)
    #cv2.waitKey(0)

'''
pathAddress=(df['image_path'])


#Loop below will through all the images in folder and extract text
for imageno in range(0,len(pathAddress)):
    #print("IMAGE NUMBER:",imageno)
    #path=getpath(Image_Folder,imageno)
    result,imageproc=OCR_Analyser(pathAddress[imageno],imageno)
    #print("The text on template is ",result)
    #df.set_value(imageno, "train", result)
    df['Output'][imageno]=result

    #print()
    #print()
    #cv2.imshow('finalimage2',imageproc)
    #cv2.waitKey(0)


df.to_csv('Outval2.csv')


