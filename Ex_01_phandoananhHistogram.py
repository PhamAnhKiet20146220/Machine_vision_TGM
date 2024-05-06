import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận
import matplotlib.pyplot as plt #Thư viện vẽ biểu đồ

#chuyển đổi ảnh mức xám bằng pp Luminance
def ChuyendoiAnhmauxambangppLumninace(imgPIL):
    #tạo ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang Grayscale
    Luminance = Image.new(imgPIL.mode, imgPIL.size)
    
    #lấy kích thước của ảnh từ imgPIL
    width = Luminance.size[0]
    height = Luminance.size[1]
    #mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 hàm for để đọc các điểm ảnh
    for x in range(width):
        for y in range(height):
            #lấy giá trị điểm ảnh tại vị trí (x,y)
            R, G, B = imgPIL.getpixel((x, y))
        
            #chuyển đổi ảnh RGB thành grayscale bằng pp Luminance
            gray = np.uint8(0.2126*R + 0.7152*G + 0.0722*B)
            
            #gán giá trị mức xám cho ảnh 
            Luminance.putpixel((x,y), (gray, gray, gray))
    return Luminance


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
def phandoan(nguong1, nguong2, tongR, tongG, tongB):
    
    # mỗi pixel có giá trị từ 0-255, nên khai báo 1 mảng có 256 phần tử để chứa số đếm 
    # của các pixel có cùng giá trị
    his = np.zeros(256)
    
    #lấy kích thước của ảnh từ imgPIL
    width = phandoanIMG.size[0]
    height = phandoanIMG.size[1]
    
    for x in range(width) :
        for y in range(height) :
            # Lấy giá trị điểm ảnh tại vị trí (a,b)
            zR,zG,zB = HinhXamPIL.getpixel((x, y))
            
            #Tính khoảng cách D theo công thức:
            #D(z,a) = SQRT[(zR - aR)^2 + (zG - aG)^2 + (zB - aB)^2]
            D = np.sqrt((zR-tongR)**2 + (zG-tongG)**2 + (zB-tongB)**2)
            
            #So sánh với GT ngưỡng để xác định z(x,y) là background hay object
            if (D >= nguong1 and D <= nguong2) :
                phandoanIMG.putpixel((x,y),(zB,zG,zR))
            #giá trị gray tính ra cũng là phần tử thứ gray trong mảng his, tăng số đếm của phần tử gray thêm 1
                his[zR] += 1
                
            else :
                phandoanIMG.putpixel((x,y),(255,255,255))
    return his


#tính histogram của ảnh xám
# def TinhHistogram(HinhXamPIL):
#     # mỗi pixel có giá trị từ 0-255, nên khai báo 1 mảng có 256 phần tử để chứa số đếm 
#     # của các pixel có cùng giá trị
#     his = np.zeros(256)
    
#     #kích thước ảnh
#     w = HinhXamPIL.size[0]
#     h = HinhXamPIL.size[1]

#     for x in range (w):
#         for y in range(h):
#             #lấy giá trị xám tại các điểm x,y
#             gR, gG, gB = HinhXamPIL.getpixel((x, y))
            
#             #giá trị gray tính ra cũng là phần tử thứ gray trong mảng his, tăng số đếm của phần tử gray thêm 1
#             his[gR] += 1
            
#     return his

#vẽ biểu đồ histogram bằng thư viện matplotlib
def VebieudoHistogram(his):
    #khai báo kích thước biểu đồ ngang là 5, cao là 4, độ phân giải là 100
    w = 5
    h = 4
    plt.figure('Biểu đồ Histogram ảnh xám', figsize=(((w, h))), dpi=100)
    trucX = np.zeros(256)  #tạo mảng 256 phần tử
    trucX = np.linspace(0, 256, 256)  #vị trí đầu 0, cuối 256
    plt.plot(trucX, his, color = 'red')
    plt.title('Biểu đồ Histogram')
    plt.xlabel('Giá trị mức xám')
    plt.ylabel('số điểm cùng giá trị mức xám')
    plt.show()

    
#chương trình chính
#khai báo đường dẫn tới file hình
filehinh = r'lena_color1.jpg'

#đọc ảnh dùng thư viện PIL
imgPIL = Image.open(filehinh)

#tạo ảnh có cùng kích thước và mode với ảnh imgPIL, để chứa kết quả chuyển đổi RGB sang Grayscale
phandoanIMG = Image.new(imgPIL.mode, imgPIL.size)

#chuyển sang mức xám
HinhXamPIL = ChuyendoiAnhmauxambangppLumninace(imgPIL)

# #tính histogram
# his = TinhHistogram(HinhXamPIL)

#gia tri x1, y1, x2, y2
Ar, Ag, Ab = vectorA(HinhXamPIL, 80, 400, 150, 500)

#ngưỡng là 0 và 45
his = phandoan(0 , 45, Ar, Ag, Ab)  

#chuyển ảnh PIL sang OpenCV để hiện thị bằng thư viện CV2
HinhXamCV = np.array(HinhXamPIL)
Hinhphandoan = np.array(phandoanIMG)

cv2.imshow('Anh Muc Xam', HinhXamCV)
cv2.imshow('Sau khi phan doan',Hinhphandoan)

#Hiển thị biểu đồ Histogram
VebieudoHistogram(his)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()
    
    
