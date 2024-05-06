import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#tạo ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang Grayscale
average = Image.new(imgPIL.mode, imgPIL.size)
lightness = Image.new(imgPIL.mode, imgPIL.size)
Luminance = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = average.size[0]
height = average.size[1]

#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp average
        gray = np.uint8((R + G + B) /3)
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp lightness
        MIN = min(R, G, B)
        MAX = max(R, G, B)
        graylightness = np.uint8((MAX + MIN) /2)
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp Luminance
        graylumi = np.uint8(0.2126*R + 0.7152*G + 0.0722*B)
        
        #gán giá trị mức xám cho ảnh 
        average.putpixel((x,y), (gray, gray, gray))
        lightness.putpixel((x,y), (graylightness, graylightness, graylightness))
        Luminance.putpixel((x,y), (graylumi, graylumi, graylumi))
        
#chuyển ảnh từ PIL sang OpenCV để hiện thị bằng OpenCV
anhmucxam1 = np.array(average)
anhmucxam2 = np.array(lightness)
anhmucxam3 = np.array(Luminance)

#hiển thị ảnh bằng openCV
cv2.imshow('Anh mau RGB goc co gai lena',img)
cv2.imshow('Anh muc xam dung Average',anhmucxam1)
cv2.imshow('Anh muc xam dung lightness',anhmucxam2)
cv2.imshow('Anh muc xam dung Luminance',anhmucxam3)


# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()