import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#lấy kích thước của ảnh
heigh = len(img[0]) #chiều cao
width = len(img[1]) #chiều ngang

#khai báo 3 biến để chứa hình 3 kênh màu RGB
Y =np.zeros((width, heigh, 3), np.uint8)  #tạo ma trận có kích thước width, height, và có ba kênh màu RGB mỗi kênh 8bit
Cr =np.zeros((width, heigh, 3), np.uint8)
Cb =np.zeros((width, heigh, 3), np.uint8)
YCrCb =np.zeros((width, heigh, 3), np.uint8)

R = img[:,:,2]
G = img[:,:,1]
B = img[:,:,0]

Y = 16 + 65.738 * R / 256 + 129.057 * G / 256 + 25.064 * B / 256
Cr = 128 - 37.945 * R / 256 - 74.494 * G / 256 + 112.439 * B / 256   
Cb = 128 + 112.439 * R / 256 - 94.154 * G / 256 - 18.285 * B / 256
           
Y = np.uint8(Y)
Cr = np.uint8(Cr)
Cb = np.uint8(Cb)

YCrCb = np.dstack((Cb, Cr, Y))

cv2.imshow('Hinh mau RGB goc co gai Lena', img)

cv2.imshow('kenh Y',Y)
cv2.imshow('kenh Cr',Cr)
cv2.imshow('kenh Cb',Cb)
cv2.imshow("kenh YCrCb",YCrCb)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()         