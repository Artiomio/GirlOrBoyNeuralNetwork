import dlib
from imread_with_EXIF_orientation import imread_with_EXIF_orientation
import math
from rotate import rotate
import sys
import os
import glob
from find_rect_range import find_rect_range



def normalized_landmark_vector(landmarks):
    """ Нормализация "волшебных" точек
        На данный момент:
            Вертикальная ориентация лица
            Приведение к единичному масштабу
    """

    # Считаем угол таким образом, что положительное направление - склонённость к правому плечу
    # Центр - 28-я точка - т.е. landmarks[27]

    nose_bridge = landmarks[27]

    eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
    angle = - math.atan(eyes_vector_y / eyes_vector_x)
    
    
    #print("Угол равен %f градусов (наклон к правому плечу)" % (angle * 180 / math.pi))
    verticalized = [rotate((x,y), origin = nose_bridge, angle = angle) for (x, y) in landmarks]

    # Временно - как хеш лица используем только глаза
    # verticalized = verticalized[42:48] + verticalized[36:42]

    ((x1, y1), (x2, y2)) = find_rect_range(verticalized)
    width = x2 - x1
    height = y2 - y1
    
 
    normalized = verticalized
    normalized = [((x-x1) / width, (y-y1) / width) for (x, y) in verticalized]
    return normalized





predictor_path = r"C:\Users\Miotio\Dropbox\Programming\python\faceArt\landmarks\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Create image window
window = dlib.image_window()


faces_folder_path = "male"

for img_file_name in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    
    try:

        img = imread_with_EXIF_orientation(img_file_name) # Считали картинку и поместили ее в ndarray
        # or use the following two lines to load an image from disk
        # from PIL import Image
        # img = Image.open(filename)
        faces, confidence, idx = detector.run(img, 1)
        # or it could be faces = detector(img, 1) - but this way face detection confidence is no longer available


        window.clear_overlay()
        window.set_image(img)


        #print("Found faces: ", len(faces))
        if len(faces)==1:
            #print("_______________________________________________________________")
            shape = predictor(img, faces[0]) # Getting landmarks (magic dots)
            window.add_overlay(shape)    # Draw the contour
            print(normalized_landmark_vector([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]) + [img_file_name])
            old_name = img_file_name
            new_name = os.path.join(faces_folder_path, "processed", os.path.split(img_file_name)[1])
            os.rename(old_name, new_name)            
                
        else:
            #print("Bad number of faces!")
            old_name = img_file_name
            new_name = os.path.join(faces_folder_path, "bad", os.path.split(img_file_name)[1])
            os.rename(old_name, new_name)

        #print("===============================================================")
    except:
        pass    

    

