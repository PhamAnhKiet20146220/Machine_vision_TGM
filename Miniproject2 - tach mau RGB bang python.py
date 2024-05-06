import cv2          #Sử dụng thư viện xử lý ảnh OpenCV 
import numpy as np  #Sử dụng thư viện numpy, đặt biệt là tính toán ma trận

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread('lena_color1.jpg', cv2.IMREAD_COLOR)

#lấy kích thước của ảnh
heigh = len(img[0]) #chiều cao
width = len(img[1]) #chiều ngang

#khai báo 3 biến để chứa hình 3 kênh màu RGB
red =np.zeros((width, heigh, 3), np.uint8)  #tạo ma trận có kích thước width, height, và có ba kênh màu RGB mỗi kênh 8bit
green =np.zeros((width, heigh, 3), np.uint8)
blue =np.zeros((width, heigh, 3), np.uint8)  

#set zero cho tất cả điểm ảnh trong cả 3 kênh ở mỗi hình
red[:] = [0,0,0]
green[:] = [0,0,0]
blue[:] = [0,0,0]

#mỗi hình là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc hết các điểm ảnh(pixel)
for x in range(width):                                  #chạy theo chiều ngang
    for y in range(heigh):                              #chạy theo chiều dọc
        #lấy giá trị điểm ảnh tại các vị trí x và y
        R = img[x,y,2]                                  #red chứa ở kênh 2
        G = img[x,y,1]                                  #green chứa ở kênh 1
        B = img[x,y,0]                                  #blue chứa ở kênh 0
        #thiết lập màu cho các kênh
        red[x,y,2] = R           #gán giá trị red cho kênh 2
        green[x,y,1] = G         #gán giá trị green cho kênh 1
        blue[x,y,0] = B          #gán giá trị blue cho kênh 0

#hiển thị hình dùng thư viện OpenCV
cv2.imshow('Hinh mau RGB goc co gai Lena', img)

cv2.imshow('kenh RED',red)
cv2.imshow('kenh GREEN',green)
cv2.imshow('kenh BLUE',blue)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()