import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận
import math

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo 4 ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi
HUE = Image.new(imgPIL.mode, imgPIL.size)
Saturation = Image.new(imgPIL.mode, imgPIL.size)
Intensity = Image.new(imgPIL.mode, imgPIL.size)
HSI = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = HUE.size[0]
height = HUE.size[1]

#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        # tử số của công thức tính theta
        tu = ((R - G) + (R - B)) / 2
        
        # mẫu số của công thức tính theta
        mau = math.sqrt((R - G) * (R - G) + (R - B) * (G - B))
        
        #kết quả hàm tính Acos trong python là radian       
        theta = math.acos(tu/mau)
        
        # công thức tính giá trị Hue
        H = 0
        
        # nếu blue <= green thì Hue = theta
        if B <= G :
            H = theta
        else: # ngược lại blue > green thì Hue:
            H = 2*math.pi - theta       # do theta tính đc ở trên là radian nên dùng Pi
            
        #chuyển đổi từ radian sang độ
        H = np.uint8(H*180/math.pi)
        
        #công thức tính giá trị kênh Saturation
        S = 1 - (3*min(R, G, B) / (R + G + B))
        
        #Do giá trị tính ra của S trong khoảng [0,1], phải convert S sang [0,255]
        S = np.uint8(S *255)
        
        #công thức tính giá trị Intensity
        I = np.uint8((R + G + B) / 3)
        
        HUE.putpixel((x,y), (H, H, H))
        
        Saturation.putpixel((x,y), (S, S, S))
        
        Intensity.putpixel((x,y), (I, I, I))
        
        #trong thư viện PIL, hàm putpixel là BGR, nên phải thay đổi vị trí
        HSI.putpixel((x,y), (I, S, H))

#chuyển ảnh từ PIL sang OpenCV để hiện thị bằng OpenCV
kenhHue = np.array(HUE)
kenhSaturation = np.array(Saturation)
kenhIntensity = np.array(Intensity)
HinhHSI = np.array(HSI)

#hiển thị ảnh bằng openCV
cv2.imshow('Anh mau RGB goc co gai lena',img)
cv2.imshow('Kenh HUE',kenhHue)
cv2.imshow('Kenh Saturation',kenhSaturation)
cv2.imshow('Kenh Intensity',kenhIntensity)
cv2.imshow('Hinh HSI',HinhHSI)


# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()