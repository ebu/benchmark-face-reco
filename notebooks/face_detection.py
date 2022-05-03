import numpy as np
from skimage.transform import resize


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x)), epsilon))
    return output

def face_cropping(image_np: np.ndarray,
                  face_detector,
                   max_nb_faces: int = 10000,
                   margin: int = 0,
                   image_size: np.uint = 160,
                   flag_normalise: bool = False,
                   flag_plot: bool = False) -> tuple[np.ndarray, np.ndarray]:

    detected = face_detector.detect_faces(image_np)
    
    if len(detected) == 0:
        
        return np.array([])
    
    if len(detected) <= max_nb_faces:
        
        faces_resize = []

        for face in detected:

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

            face_resize_np = resize(
                cropped, (image_size, image_size), mode='reflect')

            faces_resize.append(face_resize_np)
            
        faces_resize_np = np.array(faces_resize)
        return (faces_resize_np)


    else:
        
        if flag_plot:
            
            print('')
            print('-----')
            print(' {} faces detected'.format(len(detected)))
            print('-----')

            plt.figure()
            plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            plt.show()
        
        return np.array([])
    
        



