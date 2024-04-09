# General
import os
import numpy as np
#import pandas as pd
import random
import matplotlib.pyplot as plt
import math

# Images
import cv2
import dlib

def crop_faces(detector, face_detector, MARGIN, ORIGINAL_IMGS_DIR, CROPPED_IMGS_DIR, sign='pos', plot_images=False, max_images_to_plot=5, face_align=False):
    
    
    not_crop_count = 0 # count the number of faces that have not been detected by detector
    
    # We create a directory to stock the cropped images
    if not os.path.exists(CROPPED_IMGS_DIR):
        os.makedirs(CROPPED_IMGS_DIR) 
    
    # Start the cropping
    print('Cropping faces and saving to %s', CROPPED_IMGS_DIR)
    good_cropped_img_file_names = []
    original_images_detected = []
    cropped_imgs_info=[]
    it=0
    
    for file_name in sorted(os.listdir(ORIGINAL_IMGS_DIR)):
        np_img = cv2.imread(os.path.join(ORIGINAL_IMGS_DIR,file_name))
        
        
        if file_name[0:1]=='Th':
            continue
            
        # Face alignment
        if face_align==True:
            np_img=face_alignment(np_img)
        
        
        
        ######## FACE CROPPING ##########
        
        if detector=='dlib':
            gray=cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            face_rects = face_detector(gray,1)
            
            if len(face_rects) != 1:
                not_crop_count += 1
                print('NO face detected in', file_name)
                it+=1
                continue
            
            for rect in face_rects:
                x = rect.left()
                y = rect.top()
                w = rect.right() - x
                h = rect.bottom() - y
                
                if sign=='pos':
                    xw1 = max(int(x - MARGIN * (w)),0)
                    yw1 = max(int(y - MARGIN * (h)),0)
                    xw2 = min(int((x+w) + MARGIN * (w)), np.shape(gray)[1])
                    yw2 = min(int((y+h) + MARGIN * (h)), np.shape(gray)[0])

                if sign=='neg':
                    xw1 = int(x + MARGIN * (w))
                    yw1 = int(y + MARGIN * (h))
                    xw2 = int((x+w) - MARGIN * (w))
                    yw2 = int((y+h) - MARGIN * (h))
                break
                
            cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2) # We apply the cropping function defined below
            norm_file_path = '%s/%s' % (CROPPED_IMGS_DIR, file_name)
            cv2.imwrite(norm_file_path, cropped_img)
            good_cropped_img_file_names.append(file_name)
                
        if detector=='mtcnn':
                    
            img = cv2.cvtColor(cv2.imread(os.path.join(ORIGINAL_IMGS_DIR,file_name)), cv2.COLOR_BGR2RGB)
            face_rects = face_detector.detect_faces(img)
            
            for rect in face_rects:
                
                if len(face_rects) != 1:
                    not_crop_count += 1
                    print('No face detected in', file_name)
                    it+=1
                    continue
                    
                # get coordinates
                #print(rect['box'])
                x, y, w, h = rect['box']

                if sign=='pos':
                    xw1 = max(int(x - MARGIN * (w)),0)
                    yw1 = max(int(y - MARGIN * (h)),0)
                    xw2 = min(int((x+w) + MARGIN * (w)), np.shape(img)[1])
                    yw2 = min(int((y+h) + MARGIN * (h)), np.shape(img)[0])

                if sign=='neg':
                    xw1 = int(x + MARGIN * (w))
                    yw1 = int(y + MARGIN * (h))
                    xw2 = int((x+w) - MARGIN * (w))
                    yw2 = int((y+h) - MARGIN * (h))
                break
                    
                    
            cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2) # We apply the cropping function defined below

            norm_file_path = '%s/%s' % (CROPPED_IMGS_DIR, file_name)
            cv2.imwrite(norm_file_path, cropped_img)
                    
                
                
          
                
        if detector=='violajones':
                        
            img = cv2.cvtColor(cv2.imread(os.path.join(ORIGINAL_IMGS_DIR,file_name)), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face_rects = face_detector.detectMultiScale(gray, 1.3, 5)            
            for rect in face_rects:
            
                if len(face_rects) != 1:
                    not_crop_count += 1
                    print('No face detected in', file_name)
                    it+=1
                    break
                    
                # get coordinates
                #print(rect['box'])
                x, y, w, h = rect

                if sign=='pos':
                    xw1 = max(int(x - MARGIN * (w)),0)
                    yw1 = max(int(y - MARGIN * (h)),0)
                    xw2 = min(int((x+w) + MARGIN * (w)), np.shape(img)[1])
                    yw2 = min(int((y+h) + MARGIN * (h)), np.shape(img)[0])

                if sign=='neg':
                    xw1 = int(x + MARGIN * (w))
                    yw1 = int(y + MARGIN * (h))
                    xw2 = int((x+w) - MARGIN * (w))
                    yw2 = int((y+h) - MARGIN * (h))
                    
                cropped_img = crop_image(img[:,:,::-1], xw1, yw1, xw2, yw2) # We apply the cropping function defined below
                
                
                norm_file_path = '%s/%s' % (CROPPED_IMGS_DIR, file_name)
                cv2.imwrite(norm_file_path, cropped_img)
                break

            #good_cropped_img_file_names.append(file_name)
        
        # Add info of good cropped images for later
        #cropped_imgs_info.append(all_imgs_info[it])
        
        it+=1
    
    print('End. Number of not cropped faces: ', not_crop_count)
    
    return not_crop_count
                
    # Save the info of the data with cropped faces in a file
    #with open(CROPPED_IMGS_INFO_FILE, 'w') as f:
        #f.write('%s\n' % column_headers)
        #for l in cropped_imgs_info:
            #f.write('%s\n' % l)
            
            
            
            
            

def face_alignment(img):
    
    roi_color=img
    roi_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # EYES DETECTION
    # Creating variable eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    index=0
    
    if np.shape(eyes)[0]>=2:
    
        # Creating for loop in order to divide one eye from another
        for (ex , ey,  ew,  eh) in eyes:
            if index == 0:
                eye_1 = (ex, ey, ew, eh)
            elif index == 1:
                eye_2 = (ex, ey, ew, eh)
            index = index + 1
            
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
    
    # EYE CENTER COMPUTATION
    # Calculating coordinates of a central points of the rectangles
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0] 
        left_eye_y = left_eye_center[1]
 
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
    
    # ANGLE COMPUTATION
        if left_eye_y > right_eye_y:
            A = (right_eye_x, left_eye_y)
    # Integer -1 indicates that the image will rotate in the clockwise direction
            direction = -1 
        else:
            A = (left_eye_x, right_eye_y)
          # Integer 1 indicates that image will rotate in the counter clockwise  
          # direction
            direction = 1 
        
        # Angle
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle=np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
    
    # Width and height of the image
        h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
        center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
        if angle>10:
            angle=0
        if angle<-10:
            angle=0
        
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
        alig_img = cv2.warpAffine(img, M, (w, h))
        
    else:
        alig_img=img
    
    return alig_img
            

def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords