import numpy as np
from PIL import Image


'''
convert 1-d list to 2-d list  
'''
def chunks(l):
    return [l[i:i + 28] for i in range(0, len(l), 28)]


'''
multiply each element of list by 255
'''
def times(x):
    return x * 255


def generated(image):
	newimage2 = list(map(times, image))
	new2dimage = chunks(newimage2)
	array = np.array(new2dimage, dtype=np.uint8)
	new_image = Image.fromarray(array)
	new_image.save('generated_7.png')


def original(image):
	new2dimage = chunks(image)
	array = np.array(new2dimage, dtype=np.uint8)
	new_image = Image.fromarray(array)
	new_image.save('original_7.png')
