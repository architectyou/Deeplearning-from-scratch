import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])

# print(A)

cal = A.sum(axis = 0) # 수직으로 덧셈을 수행, 수직축은 0, 수평축은 1
percentage = 100*A/cal.reshape(1, 4) # A 행렬을 취한 후, 1*4 행렬로 나누어주었음. (행렬 A를 사용하는 broadcasting 예제)
# reshape 이용해서 행렬의 크기 확인하기
# print()
# print(cal)
# print(percentage)

# Note on python / Numpy Vectors

a = np.random.randn(5) # 5개의 임의의 가우스 변수가 생성됨
print(a)
print(a.shape) # 1순위 배열. (행 백터도 아니고 열 벡터도 아님) -> 모양이 n인 1순위 배열의 데이터 구조를 사용하지 않도록 한다.
print(a.T) # a와 a transpose가 일치함
print(np.dot(a, a.T))

x = np.array([[[1], [2]],
              [[3], [4]]])
print(x.shape)

a = np.random.randn(1, 3)
b = np.random.randn(3, 3)
c = a * b
print(c.shape) # it's not possible to broadcast more than one dimension

a = np.array([[2, 1], [1, 3]])
print(np.dot(a, a))

a = np.array([[1, 1], [1, -1]])
b = np.array([[2], [3]])
c = a + b
print(c)

(a * b) - (a + c) + (b * c)