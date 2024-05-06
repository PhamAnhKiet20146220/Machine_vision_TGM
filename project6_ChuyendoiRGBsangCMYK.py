import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo 4 ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang CMYK
cyan = Image.new(imgPIL.mode, imgPIL.size)
magenta = Image.new(imgPIL.mode, imgPIL.size)
yellow = Image.new(imgPIL.mode, imgPIL.size)
black = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = cyan.size[0]
height = cyan.size[1]

#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        #trong thư viện PIL, hàm putpixel là BGR, nên phải thay đổi vị trí
        #Cyan (xanh dương) là kết hợp green và blue nên red = 0
        cyan.putpixel((x,y), (B, G, 0))
        
        #Magenta (tím) là sự kết hợp giữa red và blue nên green = 0
        magenta.putpixel((x,y), (B, 0, R))
        
        #Yellow (Vàng) là sự kết hợp giữa Red và Green nên Blue = 0
        yellow.putpixel((x,y), (0, G, R))
        
        #lấy giá trị min trong 3 giá trị RGB
        MIN = min(R, G, B)
        black.putpixel((x,y), (MIN, MIN, MIN))

#chuyển ảnh từ PIL sang OpenCV để hiện thị bằng OpenCV
xanhduong = np.array(cyan)
tim = np.array(magenta)
vang = np.array(yellow)
den = np.array(black)

#hiển thị ảnh bằng openCV
cv2.imshow('Anh mau RGB goc co gai lena',img)
cv2.imshow('Kenh Cyan',xanhduong)
cv2.imshow('Kenh Magenta',tim)
cv2.imshow('Kenh Yellow',vang)
cv2.imshow('Kenh K',den)


# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()