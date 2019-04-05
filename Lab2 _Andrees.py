import numpy as np

images=np.zeros([9,400,600])
#image=np.load("E:\Facultate\IA\images\images")
for i in range(0,9):
	path='E:\Facultate\IA\images\car_'+str(i)+'.npy'
	image=np.load(path)
	images[i,:,:]=image
print(np.sum(images)) 
print(np.sum(images,axis=(1,2)))
print(np.argmax(np.sum(images), axis=0))
mean_image=np.mean(images,axis=0) 