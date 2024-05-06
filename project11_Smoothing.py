import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận

#khai báo đường dẫn file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh màu dùng thư viện OpenCV
img = cv2.imread(filehinh, cv2.IMREAD_COLOR)

#đọc ảnh màu bằng PILLOW. sử dụng để tính toán thay vì dùng opencv
imgPIL = Image.open(filehinh)

#lấy kích thước của ảnh
heigh = len(img[0]) #chiều cao
width = len(img[1]) #chiều ngang

#khai báo 4 biến để chứa hình 4 kênh màu 
smooth3x3 =np.zeros((width, heigh, 3), np.uint8)  #tạo ma trận có kích thước width, height, và có ba kênh màu RGB mỗi kênh 8bit
smooth5x5 =np.zeros((width, heigh, 3), np.uint8)
smooth7x7 =np.zeros((width, heigh, 3), np.uint8)
smooth9x9 =np.zeros((width, heigh, 3), np.uint8)


R = img[:,:,2]
G = img[:,:,1]
B = img[:,:,0]

#mỗi hình là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc hết các điểm ảnh(pixel)
for x in range(1, width - 1):
    for y in range(1, heigh - 1):
        smooth3x3[x, y, 2] = np.average(R[x - 1:x + 2, y - 1:y + 2])
        smooth3x3[x, y, 1] = np.sum(G[x - 1:x + 2, y - 1:y + 2]) // 9              
        smooth3x3[x, y, 0] = np.sum(B[x - 1:x + 2, y - 1:y + 2]) // 9
 
        
for x in range(2, width - 2):
    for y in range(2, heigh - 2):        
        smooth5x5[x, y, 2] = np.sum(R[x - 2:x + 3, y - 2:y + 3]) // 25              
        smooth5x5[x, y, 1] = np.sum(G[x - 2:x + 3, y - 2:y + 3]) // 25              
        smooth5x5[x, y, 0] = np.sum(B[x - 2:x + 3, y - 2:y + 3]) // 25
        
for x in range(3, width - 3):
    for y in range(3, heigh - 3):        
        smooth7x7[x, y, 2] = np.sum(R[x - 3:x + 4, y - 3:y + 4]) // 49              
        smooth7x7[x, y, 1] = np.sum(G[x - 3:x + 4, y - 3:y + 4]) // 49              
        smooth7x7[x, y, 0] = np.sum(B[x - 3:x + 4, y - 3:y + 4]) // 49
        
for x in range(4, width - 4):
    for y in range(4, heigh - 4):        
        smooth9x9[x, y, 2] = np.sum(R[x - 4:x + 5, y - 4:y + 5]) // 81              
        smooth9x9[x, y, 1] = np.sum(G[x - 4:x + 5, y - 4:y + 5]) // 81             
        smooth9x9[x, y, 0] = np.sum(B[x - 4:x + 5, y - 4:y + 5]) // 81
        
                            
                                

cv2.imshow('Hinh mau RGB goc co gai Lena', img)

cv2.imshow('Hinh lam muot voi mat na 3x3',smooth3x3)
cv2.imshow('Hinh lam muot voi mat na 5x5',smooth5x5)
cv2.imshow('Hinh lam muot voi mat na 7x7',smooth7x7)
cv2.imshow("Hinh lam muot voi mat na 9x9",smooth9x9)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()         