import dlib
from imread_with_EXIF_orientation import imread_with_EXIF_orientation
predictor_path = r"C:\Users\Miotio\Dropbox\Programming\python\faceArt\landmarks\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Create image window
window = dlib.image_window()



img_file_name = r"graduation2013.jpg"

img = imread_with_EXIF_orientation(img_file_name) # Считали картинку и поместили ее в ndarray
# or use the following two lines to load an image from disk
# from PIL import Image
# img = Image.open(filename)
faces, confidence, idx = detector.run(img, 1)
# or it could be faces = detector(img, 1) - but this way face detection confidence is no longer available



window.set_image(img)


print("Found faces: ", len(faces))

for (i, face) in enumerate(faces):
        print("_______________________________________________________________")
        print("Face number {}\nRectangle (x1={}, y1={}, x2={}, y2={})".format(i, face.left(), face.top(), face.right(), face.bottom()))
        print("Confidence: ", confidence[i])


        shape = predictor(img, face) # Getting landmarks (magic dots)
        window.add_overlay(shape)    # Draw the contour
        print("Magic dots coordinates: ", [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        

dlib.hit_enter_to_continue()
