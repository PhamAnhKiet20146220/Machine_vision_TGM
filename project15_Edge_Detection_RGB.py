import cv2
from PIL import Image
import numpy as np


pi = 3.14

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo  ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả sau khi chuyển đổi
anhketqua = Image.new(imgPIL.mode, imgPIL.size)


#lấy kích thước của ảnh từ imgPIL
width = anhketqua.size[0]
height = anhketqua.size[1]

#tạo ngưỡng để xét giá trị
nguong = 130

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
        gxR = 0
        gyR = 0
        
        gxG = 0
        gyG = 0
        
        gxB = 0
        gyB = 0
        for i in range(x - 1, x + 2) :
            for j in range(y - 1,y + 2) :
                R, G, B = imgPIL.getpixel((i,j))        #ảnh màu xám nên kênh RGB như nhau
                
                gxR += R*MTSobel_x[i - x + 1, j - y + 1 ]
                gyR += R*MTSobel_y[i - x + 1, j - y + 1 ]
                
                gxG += G*MTSobel_x[i - x + 1, j - y + 1 ]
                gyG += G*MTSobel_y[i - x + 1, j - y + 1 ]
                
                gxB += B*MTSobel_x[i - x + 1, j - y + 1 ]
                gyB += B*MTSobel_y[i - x + 1, j - y + 1 ]
                
        Gxx = np.abs(gxR) * np.abs(gxR) + np.abs(gxG)*np.abs(gxG) + np.abs(gxB)*np.abs(gxB)
        Gyy = np.abs(gyR) * np.abs(gyR) + np.abs(gyG)*np.abs(gyG) + np.abs(gyB)*np.abs(gyB)
        
        Gxy = gxR*gyR + gxG*gyG + gxB*gyB
        
        theta = 0.5* np.arctan2((2*Gxy),(Gxx - Gyy))
        
        if(Gxx > 0):
            theta = np.arctan(Gyy/Gxx)
        if(Gxx < 0 & Gyy >= 0):
            theta = np.arctan(Gyy/Gxx) + pi
        if(Gxx < 0 & Gyy < 0):
            theta = np.arctan(Gyy/Gxx) - pi
        if(Gxx == 0 & Gyy > 0):    
            theta = pi
        if(Gxx == 0 & Gyy < 0):
            theta = - pi
        
        F0 = np.sqrt( 0.5* ((Gxx + Gyy) + (Gxx - Gyy)*np.cos(2*theta) + 2*Gxy*np.sin(2*theta) ) )
        
        
        #So sánh M với GT ngưỡng
        if (F0 <= nguong) :
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