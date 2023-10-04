import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

from skimage.transform import resize

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x)), epsilon))
    return output

def face_cropping(image_np: np.ndarray,
                  face_detector,
                   margin: int = 0,
                   image_size: np.uint = 160,
                   min_crop_ratio : np.float32 = 0.1,
                   flag_normalise: bool = False,
                   flag_plot: bool = False) -> np.ndarray :
    

    min_size_detect = min_crop_ratio*image_np.size
    
    detected = face_detector.detect_faces(image_np)
        
    if len(detected) == 0:
        
        return np.array([])
    
    if len(detected) > 1:
        
        #keep the biggest detected face only
        area=[]
        for face in detected:
            face_box = face['box']
            area.append(face_box[2] * face_box[3])
            
        face = detected[area.index(max(area))]
        
    else:
        
        face = detected[0]

    #perform the cropping + resizing
    face_box = face['box']

    y = int(max(face_box[0] - margin/2, 0))
    x = int(max(face_box[1] - margin/2, 0))
    h = face_box[2] + margin
    w = face_box[3] + margin

    # crop the face
    if flag_normalise:
        cropped = l2_normalize(image_np[x:x + w, y:y + h, :])
    else:
        cropped = image_np[x:x + w, y:y + h, :]

    # only the face that are greater than the min_crop_ratio of the input image size are kept
    if cropped.size > min_size_detect:

        face_resize_np = resize(
            cropped, (image_size, image_size), mode='reflect')
        faces_resize = np.array([face_resize_np])

    else:
        
        print('')
        print('-----')
        print(' ration cropping achieved : ' + str((1.0*cropped.size)/image_np.size))
        print('')
        print('-----')        
        
        if flag_plot:

            print('')
            print('-----')
            print(' {} faces detected'.format(len(detected)))
            print('-----')

            plt.figure()
            plt.imshow(cropped)
            plt.show()

            plt.figure()
            plt.imshow(image_np)
            plt.show()

        faces_resize = np.array([]) 
    
    return faces_resize




