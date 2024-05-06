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

#tính histogram của ảnh xám
def TinhHistogram(HinhXamPIL):
    # mỗi pixel có giá trị từ 0-255, nên khai báo 1 mảng có 256 phần tử để chứa số đếm 
    # của các pixel có cùng giá trị
    his = np.zeros(256)
    
    #kích thước ảnh
    w = HinhXamPIL.size[0]
    h = HinhXamPIL.size[1]

    for x in range (w):
        for y in range(h):
            #lấy giá trị xám tại các điểm x,y
            gR, gG, gB = HinhXamPIL.getpixel((x, y))
            
            #giá trị gray tính ra cũng là phần tử thứ gray trong mảng his, tăng số đếm của phần tử gray thêm 1
            his[gR] += 1
    return his

#vẽ biểu đồ histogram bằng thư viện matplotlib
def VebieudoHistogram(his):
    #khai báo kích thước biểu đồ ngang là 5, cao là 4, độ phân giải là 100
    w = 5
    h = 4
    plt.figure('Biểu đồ Histogram ảnh xám', figsize=(((w, h))), dpi=100)
    trucX = np.zeros(256)  #tạo mảng 256 phần tử
    trucX = np.linspace(0, 256, 256)  #vị trí đầu 0, cuối 256
    plt.plot(trucX, his, color = 'orange')
    plt.title('Biểu đồ Histogram')
    plt.xlabel('Giá trị mức xám')
    plt.ylabel('số điểm cùng giá trị mức xám')
    plt.show()
    
#chương trình chính
#khai báo đường dẫn tới file hình
filehinh = r'bird_small.jpg'

#đọc ảnh dùng thư viện PIL
imgPIL = Image.open(filehinh)

#chuyển sang mức xám
HinhXamPIL = ChuyendoiAnhmauxambangppLumninace(imgPIL)

#tính histogram
his = TinhHistogram(HinhXamPIL)

#chuyển ảnh PIL sang OpenCV để hiện thị bằng thư viện CV2
HinhXamCV = np.array(HinhXamPIL)
cv2.imshow('Anh Muc Xam', HinhXamCV)

#Hiển thị biểu đồ Histogram
VebieudoHistogram(his)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()
    
    
