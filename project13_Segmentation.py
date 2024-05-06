import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#tạo hàm tính vector a
def vectorA(imgPIL,x1, y1, x2, y2):
    
    tongR = tongG = tongB = 0
    
    for x in range (x1, x2 + 1 ) :
        for y in range (y1, y2 + 1 ) :
            #Lấy giá trị điểm ảnh từ vị trí x1 y1 đế x2 y2
            R, G, B = imgPIL.getpixel((x,y))
            
            #Cộng dồn điểm ảnh cho mỗi kênh R-G-B tương ứng
            tongR += R
            tongG += G
            tongB += B
    #Tính kích thước vector a 
    kichthuoc = np.abs(x2+1-x1) * np.abs(y2+1-y1)
    
    tongR /= kichthuoc
    tongG /= kichthuoc
    tongB /= kichthuoc
    
    #trả về giá trị trung bình cộng của RGB sau khi tính
    return tongR, tongG, tongB

#hàm tính phân đoạn ảnh
def phandoan(nguong, tongR, tongG, tongB):
    
    #lấy kích thước của ảnh từ imgPIL
    width = phandoanIMG.size[0]
    height = phandoanIMG.size[1]
    
    for x in range(width) :
        for y in range(height) :
            # Lấy giá trị điểm ảnh tại vị trí (a,b)
            zR,zG,zB = imgPIL.getpixel((x, y))
            
            #Tính khoảng cách D theo công thức:
            #D(z,a) = SQRT[(zR - aR)^2 + (zG - aG)^2 + (zB - aB)^2]
            D = np.sqrt((zR-tongR)**2 + (zG-tongG)**2 + (zB-tongB)**2)
            #So sánh với GT ngưỡng để xác định z(x,y) là background hay object
            if (D<=nguong) :
                phandoanIMG.putpixel((x,y),(255,255,255))
            else :
                phandoanIMG.putpixel((x,y),(zB,zG,zR))
    
#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)
#tạo ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang Grayscale
phandoanIMG = Image.new(imgPIL.mode, imgPIL.size)

#gia tri x1, y1, x2, y2
Ar, Ag, Ab = vectorA(imgPIL, 80, 400, 150, 500)

#ngưỡng là 45
phandoan(45, Ar, Ag, Ab)        
      
Hinhphandoan = np.array(phandoanIMG)

cv2.imshow('Hinh mau RGB goc co gai Lena', img)
cv2.imshow('Sau khi phan doan',Hinhphandoan)


# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()         