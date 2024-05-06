import cv2
from PIL import Image
import numpy as np

#khai báo đường dẫn file hình
filehinh = r'ktx1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo 2 ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang Grayscale
# và chứa kết quả sau khi chuyển đổi
average = Image.new(imgPIL.mode, imgPIL.size)
anhketqua = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = average.size[0]
height = average.size[1]

#tạo ngưỡng để xét giá trị
nguong = 50

        
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp average
        gray = np.uint8((R + G + B) /3)
        
        #gán giá trị mức xám cho ảnh 
        average.putpixel((x,y), (gray, gray, gray))

#tạo ma trận Sobel
#Theo phương x
MTSobel_x = np.array([[-1,-2,-1],
                      [ 0, 0, 0],
                      [ 1, 2, 1]])

#Theo phương y
MTSobel_y = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

#Nhân tích chập các ma trận theo ma trận Sobel
for x in range(1, width - 1) :
    for y in range(1, height - 1) :
        gx = 0
        gy = 0
        for i in range(x - 1, x + 2) :
            for j in range(y-1,y+2) :
                GraySobel, GraySobel, GraySobel = average.getpixel((i,j))        #ảnh màu xám nên kênh RGB như nhau
                
                gx += GraySobel*MTSobel_x[i - x + 1, j - y + 1 ]
                gy += GraySobel*MTSobel_y[i - x + 1, j - y + 1 ]
                
        #Tính biên độ M của vector theo công thức trị tuyệt đối Gx cộng Gy
        M = np.abs(gx) + np.abs(gy)
        
        #So sánh M với GT ngưỡng
        if (M <= nguong) :
            anhketqua.putpixel((x,y),(0,0,0))
        else :
            anhketqua.putpixel((x,y),(255,255,255))

#Chuyển ảnh từ PIL sang OpenCV để hiển thị bằng OpenCV
anhnhandang = np.array(anhketqua)

#Hiển thị ảnh với OpenCV
cv2.imshow('Anh mau RGB goc co gai lena', img)
cv2.imshow('nhan dang bien anh xam bang pp sobel', anhnhandang)

#Bấm phím bất kì để đóng cửa sổ
cv2.waitKey()

#Giải phóng bộ nhớ
cv2.destroyAllWindows()