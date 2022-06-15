from __future__ import print_function

import binascii

import numpy as np
import scipy.cluster
from PIL import Image

NUM_CLUSTERS = 15

print('reading image')
im = Image.open('A S T R O - Change.jpg')
im = im.resize((150, 150))  # optional, to reduce time
ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
print('cluster centres:\n', codes)

vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
counts, bins = np.histogram(vecs, len(codes))  # count occurrences

index_max = np.argmax(counts)  # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('most frequent is %s (#%s)' % (peak, colour))

# bonus: save image using only the N most common colours
import imageio

print("====")
print([tuple(c) for c in codes])
print("====")
print(len(vecs))
print(vecs)
print("====")

c = ar.copy()
for i, code in enumerate(codes):
    c[np.r_[np.where(vecs == i)], :] = code
imageio.imwrite('clusters.png', c.reshape(*shape).astype(np.uint8))
print('saved clustered image')
