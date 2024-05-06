import cv2                      # Sử dụng thư viện OpenCV
from PIL import Image           # Thư viện sử lý ảnh Pillow hỗ trợ nhiều loại định dạng ảnh
import numpy as np              # Thư viện toán học, ma trận

# Khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

# Đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

# Đọc ảnh màu bằng Pillow, sử dụng để tính toán thay vì dùng OpenCV
imgPIL = Image.open(filehinh)

# Lấy kích thước của ảnh
height, width, _ = img.shape  # Sửa lỗi trong cách lấy kích thước

sharpness = Image.new(imgPIL.mode, imgPIL.size)

# Tạo ma trận 3x3
matrix = [[0, -1, 0],
          [-1, 4, -1],
          [0, -1, 0]]


# Mỗi hình là 1 ma trận 2 chiều nên dùng 2 vòng for để đọc hết các điểm ảnh (pixel)
for x in range(1, width - 1):
    for y in range(1, height - 1):
        Rs = 0
        Gs = 0
        Bs = 0

        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                R, G, B = imgPIL.getpixel((i, j))

                Rs += R * matrix[i - x + 1][j - y + 1]
                Gs += G * matrix[i - x + 1][j - y + 1]
                Bs += B * matrix[i - x + 1][j - y + 1]

        Rm, Gm, Bm = imgPIL.getpixel((x, y))

        tongR = Rm + Rs
        tongG = Gm + Gs
        tongB = Bm + Bs
        
        if tongR >= 255:
            tongR = 255
        if tongG >= 255:
            tongG = 255
        if tongB >= 255:
            tongB = 255

        if tongR <= 0:
            tongR = 0
        if tongG <= 0:
            tongG = 0
        if tongB <= 0:
            tongB = 0
            
        sharpness.putpixel((x,y), (tongB, tongG, tongR))

Hinhlamnet = np.array(sharpness)

cv2.imshow('Anh mau RGB goc co gai lena', img)
cv2.imshow('Hinh lam net', Hinhlamnet)

# Bấm phím bất kì để đóng cửa sổ hiển thị hình
cv2.waitKey(0)

# Giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()
