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
lightness = Image.new(imgPIL.mode, imgPIL.size)

#lấy kích thước của ảnh từ imgPIL
width = lightness.size[0]
height = lightness.size[1]

#mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí (x,y)
        R, G, B = imgPIL.getpixel((x, y))
        
        #chuyển đổi ảnh RGB thành grayscale bằng pp lightness
        MIN = min(R, G, B)
        MAX = max(R, G, B)
        gray = np.uint8((MAX + MIN) /2)
        
        #gán giá trị mức xám cho ảnh 
        lightness.putpixel((x,y), (gray, gray, gray))
        
#chuyển ảnh từ PIL sang OpenCV để hiện thị bằng OpenCV
anhmucxam = np.array(lightness)

#hiển thị ảnh bằng openCV
cv2.imshow('Anh mau RGB goc co gai lena',img)
cv2.imshow('Anh muc xam dung lightness',anhmucxam)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()