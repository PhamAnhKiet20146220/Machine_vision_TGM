import cv2
import numpy as np
from PIL import Image

# Khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

# Đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

# Đọc ảnh màu bằng PILLOW, sử dụng để tính toán thay vì dùng OpenCV
imgPIL = Image.open(filehinh)

#tạo ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang nhị phân
Binary = Image.new(imgPIL.mode, imgPIL.size)


#lấy kích thước của ảnh từ imgPIL
width = Binary.size[0]
height = Binary.size[1]


# Thêm hàng top và bot với giá trị là 0
top_rows = 1   # Số hàng top muốn thêm
bot_rows = 1   # Số hàng bot muốn thêm
left_row = 1   # số hàng trái muốn thêm
right_row = 1  # số hàng phải muốn thêm

# Tạo ma trận 3x3
matrix_3x3 = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

# chuyển ảnh sang dạng nhị phân

#ngưỡng chuyển đổi
Nguong = 130
#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp Luminace
        gray = np.uint8(0.2126*R + 0.7152*G + 0.0722*B)
        
        #Xác định giá trị điểm nhị phân
        if (gray < Nguong):
            Binary.putpixel((x,y), (0, 0, 0))
        else:
            Binary.putpixel((x,y), (255, 255, 255))

#đánh padding cho ảnh theo thứ tự: trên dưới - trái phải - các kênh
padded_img = np.pad(Binary, ((top_rows, bot_rows), (left_row, right_row), (0, 0)), mode='constant', constant_values=0)

#lấy kích thước ảnh cho ma trận 3x3
height_img = padded_img.shape[0]
width_img = padded_img.shape[1]


# Tạo một ma trận mới để lưu kết quả Erosion và Dilation của ma trận 1x3
erosion_result =np.zeros((height_img, width_img, 3), np.uint8)
dilation_result =np.zeros((height_img, width_img, 3), np.uint8)

# tạo hàm for để AND và OR 2 ma trận 3x3
for i in range(1, height_img - 1):
    for j in range(1, width_img - 1):
        # Lấy phần tương ứng với ma trận 3x3
        ma_tran_a = padded_img[i - 1 : i + 2, j - 1 : j + 2]
        
       # Áp dụng phép toán Erosion
        if np.all(ma_tran_a[matrix_3x3 == 1] == 255):
            erosion_result[i, j] = 255
        else:
            erosion_result[i, j] = 0
        
        # Áp dụng phép toán Dilation
        result = ma_tran_a * matrix_3x3
        if np.any(result == 255):
            dilation_result[i, j] = 255
        else:
            dilation_result[i, j] = 0
        

# Hiển thị ảnh
# cv2.imshow('Anh RGB goc', img)
cv2.imshow('anh binary', padded_img)
cv2.imshow('Erosion (AND) ma tran 3x3', erosion_result)
cv2.imshow('Dilation (OR) ma tran 3x3', dilation_result)


cv2.waitKey(0)
cv2.destroyAllWindows()
