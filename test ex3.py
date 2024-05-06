# import numpy as np

# # Ma trận 1x3
# matrix_1x3 = np.array([[1],
#                        [1],
#                        [1]])

# # Hình ảnh mà bạn muốn áp dụng Erosion
# image = np.array([[255, 255, 255, 255, 0],
#                   [255, 255, 0, 255, 255],
#                   [255, 255, 255, 255, 255],
#                   [0, 255, 255, 255, 0],
#                   [0, 255, 255, 255, 0]])

# # Kích thước ảnh
# rows, cols = image.shape

# # Tạo một ma trận mới để lưu kết quả Erosion
# erosion_result = np.zeros_like(image)

# # Áp dụng Erosion
# for i in range(1, rows - 1 ):
#     for j in range(cols):
#         # Lấy phần tương ứng với ma trận 1x3
#         neighborhood = image[i-1:i+2, j]

#         # Áp dụng phép toán Erosion
#         erosion_result[i, j] = np.min(np.multiply(neighborhood, matrix_1x3))

# # Hiển thị kết quả Erosion
# print("Ma trận ảnh gốc:")
# print(image)
# print("\nKết quả Erosion:")
# print(erosion_result)


#------------------------------------------------------------------------1x3

# import numpy as np

# # Ma trận 1x3
# matrix_1x3 = np.array([[1, 1, 1]])

# # Hình ảnh mà bạn muốn áp dụng Erosion
# image = np.array([[255, 255, 255, 255, 0],
#                   [0, 255, 0, 255, 255],
#                   [0, 255, 255, 255, 255],
#                   [0, 255, 255, 0, 0],
#                   [0, 255, 255, 255, 0]])

# # Kích thước ảnh
# rows, cols = image.shape

# # Tạo một ma trận mới để lưu kết quả Erosion
# erosion_result = np.zeros_like(image)

# # Áp dụng Erosion
# for i in range(rows):
#     for j in range(cols - 2):
#         # Lấy phần tương ứng với ma trận 1x3
#         neighborhood = image[i, j : j + 3]
#         print(neighborhood)
        
#         # tạo ma trận lưu kết quả sau khi nhân 2 ma trận
#         result = np.multiply(neighborhood, matrix_1x3)
        
#         # Áp dụng phép toán Erosion
        
#         if np.any([result[0, 0] == 0, result[0, 2] == 0 ]):
#             erosion_result[i, j + 1] = 0
#         else:
#             erosion_result[i, j + 1] = 255

# # Hiển thị kết quả Erosion
# print("Ma trận ảnh gốc:")
# print(image)
# print("\nKết quả Erosion:")
# print(erosion_result)

#--------------------------------------------------3x3

import numpy as np

image = np.array([[0  , 0  , 0  , 0  , 0  ],
                  [0  , 255, 255, 255, 0  ],
                  [0  , 255, 255, 255, 0  ],
                  [0  , 255, 255, 255, 0  ],
                  [0  , 0  , 0  , 0  , 0  ]])

image1 = np.array([[0 , 0  , 0  , 0  , 0  ,0  ,0],
                  [0 , 0  , 0  , 0  , 0  , 0  ,0],
                  [0 , 0  , 255, 255, 255, 0  ,0],
                  [0 , 0  , 255, 255, 255, 0  ,0],
                  [0 , 0  , 255, 255, 255, 0  ,0],
                  [0 , 0  , 0  , 0  , 0  , 0  ,0],
                  [0 , 0  , 0  , 0  , 0  , 0  ,0]])
# Tạo ma trận 3x3
matrix_3x3 = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

# Kích thước ảnh
rows, cols = image.shape
rows1, cols1 = image1.shape
# Tạo một ma trận mới để lưu kết quả Erosion
erosion_result = np.zeros_like(image)
dilation_result = np.zeros_like(image1)
# Áp dụng Erosion
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        # Lấy phần tương ứng với ma trận 1x3
        neighborhood = image[i - 1 : i + 2, j - 1 : j + 2]
        
        # Áp dụng phép toán Erosion
        if np.all(neighborhood[matrix_3x3 == 1] == 255):
            erosion_result[i, j] = 255
        else:
            erosion_result[i, j] = 0

# Áp dụng Dilation
for i in range(1, rows1 - 1):
    for j in range(1, cols1 - 1):       
         # Áp dụng phép toán Dilation
        neighborhood1 = image1[i - 1 : i + 2, j - 1 : j + 2]    
        result = neighborhood1 * matrix_3x3
        if np.any(result == 255):
            dilation_result[i, j] = 255
        else:
            dilation_result[i, j] = 0

        
# Hiển thị kết quả Erosion
print("Ma trận ảnh gốc:")
print(image)
print("\nKết quả Erosion:")
print(erosion_result)
print("\nKết quả Erosion:")
print(dilation_result)