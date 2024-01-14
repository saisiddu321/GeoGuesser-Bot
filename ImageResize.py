from PIL import Image
import os

input_path = []
i = 0

# img = Image.open('C:/Users/sidst/OneDrive/Desktop/GeoGuesser/kagglegeogusser/Aland/canvas_1629480180.jpg')
# new_image = img.resize((500, 500))
# new_image.save('C:/Users/sidst/OneDrive/Desktop/GeoGuesser/newgeoguesser/canvas_1629480180.jpg', '')

for c in os.listdir("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/kagglegeogusser"):
    img_folder = 'newgeoguesser2/'+c
    os.mkdir(img_folder)
    for path in os.listdir("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/kagglegeogusser/"+c):
        input_path.append(os.path.join("C:/Users/sidst/OneDrive/Desktop/GeoGuesser/kagglegeogusser", c, path))
        image = Image.open(input_path[i])
        new_image = image.resize((256, 256))
        address = 'C:/Users/sidst/OneDrive/Desktop/GeoGuesser/newgeoguesser2/' +  c + '/' + path
        print(address)
        new_image.save(address, '')
        i += 1
print("done")
