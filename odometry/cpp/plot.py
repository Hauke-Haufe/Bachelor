import matplotlib.pyplot as plt
import numpy as np

image = np.load("odometry/cpp/image.npy")
plt.imshow(image)
plt.colorbar()
plt.show()

imagem = np.load("odometry/cpp/imagemaskout.npy")
plt.imshow(imagem)
plt.colorbar()
plt.show()

"""maskm = np.load("odometry/cpp/down_mask.npy")
plt.imshow(maskm)
plt.show()

mask = np.load("odometry/cpp/mask.npy")
plt.imshow(mask)
plt.show()"""