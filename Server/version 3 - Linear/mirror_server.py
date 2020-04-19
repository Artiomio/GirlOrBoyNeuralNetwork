import os
import math
from flask import Flask, request, redirect, url_for, make_response
import numpy as np
from werkzeug import secure_filename
from rotate import rotate
from find_rect_range import find_rect_range
import dlib
from imread_with_EXIF_orientation import imread_with_EXIF_orientation
from feedforward import feedforward

predictor_path = r"C:\Users\Miotio\Dropbox\Programming\python\faceArt\landmarks\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

weights = [[[17.095857256201697, 3.9743596002703563, -31.830712178523374, 7.833429551846947, 13.811223021459712, -9.22868840152572, -3.516767465084474, 11.117890022769673, 11.131754642263413, 6.191104156364622, -11.767680794445077, -9.463196859796694, 2.5635199209483663, 2.5994248559605957, 7.452221817413651, 21.241152519070546, -2.9337177281992943, -39.276547276294494, 17.986292619724797, 11.62952576787584, 4.615389263995445, -1.0992979953571094, -16.717642871756503, 1.1836447946441473, 11.54275811280459, 7.983616907544699, 7.911506249415021, -7.507052350115697, 1.367781356289699, 6.008371470279623, -31.71686299247796, 1.777728318609259, 22.505369191343952, 5.869984715877966, -6.0652045937208445, -28.167194878470358, 21.297893028243998, 10.279051236090831, 10.749451466951731, -14.840896252854584, -3.1359762611045983, -2.0941205353670624, -23.80317791425785, 26.876832790227024, -32.17857171280553, 35.28412572907156, 0.906802689441772, -7.984665152049877, 0.409391829254151, -14.212967934744796, 26.057826902348435, 4.699890437478163, -2.1696394443698876, -27.285075110341666, -0.12308356634506543, -18.823002206351237, -14.179522249700621, -2.0826305311142983, -9.149069033611754, -9.098304587950768, -10.853945645019568, -9.297330766890333, 31.69112557581736, 38.51575985516813, -5.995849716188952, -2.704498878045186, -19.836468290844522, -17.957063774641192, -4.181562092474039, 4.518787749214524, 27.743265546758703, 28.91904378566397, -12.866319545715326, 17.340159913147843, -1.0377069738598512, -4.723478792302001, -6.695676611028729, -12.841296704114214, -6.149441686813465, 21.04916771557761, 4.526956999467175, -12.148789199132096, 9.57978946592392, -4.880701404257967, 0.3007709751881192, 16.91906565398909, -8.095553208152017, -16.97471528676422, -11.84782125237844, -1.483545533206609, -30.249538733409292, 23.915395071052128, 15.497558305299792, 5.453474085443049, 12.46378748728065, -10.558247067602741, -15.93104177554988, -8.39130588128824, 4.907548432721523, -6.818747294967567, 1.5137821255092105, -2.041039628537485, -19.599383699057306, -38.462867051278806, -0.5868205033602217, -4.577372651639328, 15.964432548810635, 3.4009437138524103, -7.5401493722897195, -10.621709541232311, -1.1520516706068367, -4.737036665496696, 7.74361372062057, 19.613264646051046, -18.730998347021863, 0.06384445840506289, 12.453655940616033, 5.817758647213267, -12.286008748387474, -8.427770864203035, 24.00643850610813, 4.630686064078786, 4.5026847520527005, 20.915708194077858, -26.930991076673738, -20.101434233004646, -0.6614188377316976, 18.99229922887899, 11.176006121616231, -9.367208594199795, -0.3077587198992797, 9.6202784267279, -9.588653875349175, -15.251587455810313, 9.742121117280648, 16.2964102137917, -7.63369911971633]], [[0.8535851761717382, 0.8882273739726144]]]

static_submit_form =  "<html><body>" + open("static_html.template","r").read()

print("Привет!")
UPLOAD_FOLDER = './img/'
ALLOWED_EXTENSIONS = set(['jpg','jpeg','png'])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
		   

def distance(x1,y1, x2, y2):
	return math.sqrt( (x1-x2)**2 + (y1-y2)**2)

def magic_distances_from_landmarks(a):


	# Первая опорная точка
	xc1 = a[27][0]
	yc1 = a[27][1]


	# Вторая опорная точка
	xc2 = a[57][0]
	yc2 = a[57][1]


	distances = []

	for (x, y) in a:
		"""
		x.append(i[0])
		y.append(i[1])
		"""
		

		d1 = distance(x,y, xc1, yc1) 
		d2 = distance(x,y, xc2, yc2) 
		
		distances.append(d1)
		distances.append(d2)
	return distances




@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		print("Заголовки request.headers: ", request.headers)
		file = request.files['file']
		print("Файлы request.files=%s" % request.files)
		if file and allowed_file(file.filename):
			print("Сохраняем файл")
			filename = secure_filename(file.filename)
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(full_filename)
			print("Сохранили файл")


			img = imread_with_EXIF_orientation(full_filename) # Считали картинку и поместили ее в ndarray
			faces, confidence, idx = detector.run(img, 1)

			shape = predictor(img, faces[0])
			magic_dots = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]
			print("_______________________\n", magic_dots, "\n_____________________")
			distances = magic_distances_from_landmarks(normalized_landmark_vector(magic_dots))
			result = feedforward(distances, weights)
			print("Result: ", result)
			return "<h2> Result: " + str(result) + "</h2>"+ static_submit_form


		else:
			return '{"Error":"123","ErrorText":"Bad file format"}'
	#return  "Тут мы ждем файлик!"
	return static_submit_form
	

print("Сейчас будем запускать сервер!")
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=7777)