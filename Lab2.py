import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def main():
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    #print(y)
    plt.plot(x, y,'o')
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    # Adauga titlu
    plt.title('Sine')
    # Adauga legenda
    plt.legend(['Sine'])
    #plt.show()

    images = np.zeros([9, 400, 600])
    #a
    for i in range(0,9):
        img = np.load("E:\Facultate\IA\images\images\car_" + str(i) +".npy")
        images[i, :, :] = img
    #print(np.sum(images))

    #b
    sums = np.sum(images,axis=(1,2))
    #print(np.sum(images, axis=(1,2)))

    #c argmaxim
    sumMax = np.max(sums);
    for i in range(0,9):
        if sums[i] == sumMax:
            print(i)
    #sau
    print(np.argmax(sums))


    #d
    #for imgg in images:
     #   io.imshow(imgg.astype(np.uint8))  # petru a putea fi afisata
                                            # imaginea trebuie sa aiba
                                            # tipul unsigned int
        #io.show()

    #e imaginea medie
    mean_image = np.mean(images, axis=0)
    #print(mean_image)

    #f deviatia standard
    deviatia_standard = np.std(images)
    #print(bla)

    #g normalizarea imaginilor
    for img in images:
        img2 = np.divide(np.subtract(img,mean_image),deviatia_standard)
        #print(img2)

    #h decupare img
    for img in images:
        img2 = np.copy(img)
        print(img2[200:300,280:400])


if __name__ == "__main__":
    main()
