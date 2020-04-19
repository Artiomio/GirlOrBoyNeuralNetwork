from skimage.draw import circle       # Для рисования окружностей
def draw_circle(arr, x, y, r, color = [255, 0, 0]):
    try:
        rr, cc = circle(y, x, r)
        arr[rr, cc] = color
    except:
        print("Wrong circle!")
