import math

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












f = open("female_magic_dots", "r").readlines()
g = open("male_magic_dots", "r").readlines()

print("loaded from file!")

list_girls = [eval(x)[:-1] for x in f]
print("girls processed")


list_guys = [eval(x)[:-1] for x in g]
print("guys processed")

# Перевели текстовое представление из файла в список

print("Creating file 'magic_distances_girls'")

with open("magic_distances_girls", "w") as f:
    for x in list_girls:
        d = magic_distances_from_landmarks(x)
        if len(d) != 68*2:
            print("Wow! Something's wrong!")
        f.write(" ".join([ str(i) for i in d]))
        f.write("\n")

print("First file created!")

print("Creating file 'magic_distances_guys'")
with open("magic_distances_guys", "w") as f:
    for x in list_guys:
        d = magic_distances_from_landmarks(x)
        if len(d) != 68*2:
            print("Wow! Something's wrong!")        
        f.write(" ".join([ str(i) for i in d]))
        f.write("\n")





#s = eval(f[10860])
