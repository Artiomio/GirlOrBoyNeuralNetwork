""" Playing with beauty
"""
import math
import dlib
import numpy as np
from skimage import io
from skimage.draw import circle
from scipy.spatial import ConvexHull  # Для подсчета площади

from draw_circle import draw_circle
from rotate import rotate
from find_rect_range import find_rect_range
from imread_with_EXIF_orientation import imread_with_EXIF_orientation
import biddy

def polygon_area(poly):
    return ConvexHull(poly).volume


watch = biddy.Biddy()    

predictor_path = "shape_predictor_68_face_landmarks.dat"

watch.start()
print("Инициализация распознавателя")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
print("Распознаватель dlib.shape_predictor инициализирован (за %f сек)" % watch.end())

def get_landmarks(img):
    """ Возвращает массив из 68 волшебных точек лица
        Если лиц не обнаружено, то возвращается пустой список
    """
    #TODO Учесть степень достоверности, которая записана как одно из полей в dets
    print("Распознаём лица")
    dets = detector(img, 1)
    print("Готово, найдено лиц: %d" % len(dets))

    if len(dets) > 1:
        print("Несколько лиц! Рассматривать будем первое.")
        #return []

    if len(dets) == 0:
        print("Ошибка: Не обнаружены лица на фотографии!")
        return []

    d = dets[0]
    
    print("Запускается predictor")
    shape = predictor(img, d)
    print("Предиктор вернул результат")
    return [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]


def get_attractiveness_from_landmarks(coords):
    #TODO Возможно, стоит переделать в словарь: e.g. feature["left_eye"]=coords[42:48]
    left_eye = coords[42:48]
    right_eye = coords[36:42]
    right_brow = coords[17:22]
    left_brow = coords[22:27]
    nose_line = coords[27:31]
    nostrils = coords[31:36]
    total_nose = nostrils + [coords[27]]
    upper_lip = coords[48:55] + coords[64:59:-1]
    lower_lip = coords[54:60] + [coords[48], coords[60], coords[67], coords[66], coords[65], coords[64], coords[54]]
    face_contour = coords[0:17]
    left_eye_region = coords[22:27] + [coords[45], coords[46], coords[47], coords[42]]
    right_eye_region = coords[17:22] + [coords[39], coords[40], coords[41], coords[36]]

    eyes_area = polygon_area(left_eye) + polygon_area(right_eye)
    face_contour_area = polygon_area(face_contour)
    eyes_region_area = polygon_area(left_eye_region) + polygon_area(right_eye_region)
    nostrils_area = polygon_area(nostrils)
    total_nose_area = polygon_area(total_nose)
    lips_area = polygon_area(lower_lip) + polygon_area(upper_lip)


    detailed_result = {"face_contour_area": "%.2f" % (100 * face_contour_area / face_contour_area),
                       "eyes_area": "%.3f" % (100 * eyes_area / face_contour_area),
                       "eyes_region_area": "%.3f" % (100 * eyes_region_area / face_contour_area),
                       "nostrils_area": "%.3f" % (100 * nostrils_area / face_contour_area),
                       "total_nose_area": "%.3f" % (100 * total_nose_area / face_contour_area),
                       "lips_area": "%.3f" % (100 * lips_area / face_contour_area)
                      }

    print("Возвращаем оценку")
    return detailed_result

def normalized_landmark_vector(landmarks):
    """ Нормализация "волшебных" точек
        На данный момент:
            Вертикальная ориентация лица
            Приведение к единичному масштабу
    """

    # Считаем угол таким образом, что положительное направление - склонённость к правому плечу
    # Центр - 28-я точка - т.е. landmarks[27]
    print("Нормализация")

    nose_bridge = landmarks[27]

    eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
    angle = - math.atan(eyes_vector_y / eyes_vector_x)
    
    
    print("Угол равен %f градусов (наклон к правому плечу)" % (angle * 180 / math.pi))
    verticalized = [rotate((x,y), origin = nose_bridge, angle = angle) for (x, y) in landmarks]

    # Временно - как хеш лица используем только глаза
    # verticalized = verticalized[42:48] + verticalized[36:42]

    ((x1, y1), (x2, y2)) = find_rect_range(verticalized)
    width = x2 - x1
    height = y2 - y1
    
 
    normalized = verticalized
    normalized = [((x-x1) / width, (y-y1) / width) for (x, y) in verticalized]
    print("Размер области лица %dx%d" % (height, width))


    return normalized

def landmark_difference(landmarks_1, landmarks_2):
    return (np.sqrt(np.square(np.array(landmarks_1) -  np.array(landmarks_2)).sum(axis=1))).sum()


def face_vector_diff(face_vector_1, face_vector_2):
    """ Разница двух векторов лиц
        пока в простой реализации - просто сумма модулей разностей координат,
        т.е. taxi-cab distance НЕТ ЛУЧШЕ СДЕЛАТЬ СУММУ РАССТОЯНИЙ МЕЖДУ ТОЧКАМИ

    """
    
def assess_face_appeal(filename):
    """ Оценивает привлекательность лица из файла с изображением:
        Вот, что делает функция на данный момент:
          1) Считывает изображение из файла в массив.
          2) Получает массив "волшебных" точек.
          3) Сохраняет изображение с отмеченными "волшебными точками" и
             изображение с белым прямоугольник и отмеченными на нём "волшебными" точками
             с теми же координатами, что и в первом изображении.
          4) Вызывает функцию get_attractiveness_from_landmarks и возвращает результатом
             полученные данные.
    """
    img = imread_with_EXIF_orientation(filename) # Считали картинку и поместили ее в ndarray
    landmarks = get_landmarks(img) # Считали массив "волшебных" точек
    if not landmarks:
        return {"Error":"Face not found", "File": filename}

    print(landmarks)

    only_dots_img = np.zeros((512,512,3), dtype=np.uint8)
    
    # Сохраняем последнее изображение с кружочками на местах "волшебных" точек
    for (x, y) in landmarks:
        draw_circle(img, x, y, 3)
        
    
    for (x,y) in normalized_landmark_vector(landmarks):
        draw_circle(only_dots_img, round(x * 300 + 100), round(300 * y + 100), 3)

        
    io.imsave("./images/lastimage.png", img)
    io.imsave("./images/only_dots.png", only_dots_img)

    return get_attractiveness_from_landmarks(landmarks)

#assess_face_appeal("face_example1.jpg")
#assess_face_appeal("face_example.jpg")
#assess_face_appeal("face_example3.jpg")

    