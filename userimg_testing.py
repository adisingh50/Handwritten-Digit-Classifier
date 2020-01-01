import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as pt

user_img = np.invert(Image.open('user_num.png').convert('L')).ravel()
user_arr = np.array(user_img)
user_arr.shape

#resizing the image to scale down
resized_img = user_arr.imresize(user_arr, .1, interp= 'nearest', mode='L')

n = resized_img
n.shape = (28, 28)
pt.imshow(255 - n, cmap= 'gray')
pt.show()

#predict what number the image represemts
clf = pickle.load(open('nr_model', 'rb'))
print('Prediction: ', np.squeeze(clf.predict([resized_img])))
