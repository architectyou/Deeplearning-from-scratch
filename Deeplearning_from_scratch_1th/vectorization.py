import numpy as np

# a = np.array([1, 2, 3, 4])
# print(a)

import time # 시간 측정에 사용되는 라이브러리

a = np.random.rand(1000000) #랜덤 값을 갖는 100만 차원의 배열 생성
b = np.random.rand(1000000)

# vectorized test
tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(c)
print("Vectorized Version : " + str(1000*(toc-tic)) + "ms")


# non-vectorized(for loop version) test
c = 0
tic = time.time()
for i in range(1000000) :
    c += a[i] * b[i]
toc = time.time()
print(c)
print("For loop Version : " + str(1000*(toc-tic)) + "ms")