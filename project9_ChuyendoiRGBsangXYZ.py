import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo 4 ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi
X = Image.new(imgPIL.mode, imgPIL.size)
Y = Image.new(imgPIL.mode, imgPIL.size)
Z = Image.new(imgPIL.mode, imgPIL.size)
XYZ = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = X.size[0]
height = X.size[1]

#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
       #Tính X-Y-Z
        giatriX = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B)
        giatriY = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B)
        giatriZ = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B)

        #Hiển thị các kênh giá trị X-Y-Z
        X.putpixel((x,y),(np.uint8(giatriX),np.uint8(giatriX),np.uint8(giatriX)))
        Y.putpixel((x,y),(np.uint8(giatriY),np.uint8(giatriY),np.uint8(giatriY)))
        Z.putpixel((x,y),(np.uint8(giatriZ),np.uint8(giatriZ),np.uint8(giatriZ)))

        #Hiển thị hình XYZ tổng hợp
        XYZ.putpixel((x,y),(np.uint8(giatriZ),np.uint8(giatriY),np.uint8(giatriX)))

#chuyển ảnh từ PIL sang OpenCV để hiện thị bằng OpenCV
kenhHue = np.array(X)
kenhSaturation = np.array(Y)
kenhIntensity = np.array(Z)
HinhHSI = np.array(XYZ)

#hiển thị ảnh bằng openCV
cv2.imshow('Anh mau RGB goc co gai lena',img)
cv2.imshow('Kenh X',kenhHue)
cv2.imshow('Kenh Y',kenhSaturation)
cv2.imshow('Kenh Z',kenhIntensity)
cv2.imshow('Hinh XYZ',HinhHSI)


# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()