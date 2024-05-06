import cv2                      #sử dụng thư viện openCV
from PIL import Image           #thư viện sử lý ảnh PILLOW hỗ trợ nhiều loại định dạng ảnh
import numpy as np              #Thư viện toán học, ma trận
import matplotlib.pyplot as plt #Thư viện vẽ biểu đồ

def TinhHistogram(HinhRGB):
    # mỗi pixel có giá trị từ 0-255, nên khai báo 3 mảng có 256 phần tử để chứa số đếm 
    # của các pixel có cùng giá trị
    hisR = np.zeros(256)
    hisG = np.zeros(256)
    hisB = np.zeros(256)
    #kích thước ảnh
    w = HinhRGB.size[0]
    h = HinhRGB.size[1]

    for x in range (w):
        for y in range(h):
            #lấy giá trị xám tại các điểm x,y
            gR, gG, gB = HinhRGB.getpixel((x, y))
            
            #giá trị gray tính ra cũng là phần tử thứ gray trong mảng his, tăng số đếm của phần tử gray thêm 1
            hisR[gR] += 1
            hisG[gG] += 1
            hisB[gB] += 1
    return hisR, hisG, hisB

#vẽ biểu đồ histogram bằng thư viện matplotlib
def VebieudoHistogram(hisR, hisG, hisB):
    #khai báo kích thước biểu đồ ngang là 5, cao là 4, độ phân giải là 100
    w = 5
    h = 4
    plt.figure('Biểu đồ Histogram ảnh xám', figsize=(((w, h))), dpi=100)
    trucX = np.zeros(256)  #tạo mảng 256 phần tử
    trucX = np.linspace(0, 256, 256)  #vị trí đầu 0, cuối 256
    plt.plot(trucX, hisR, color = 'Red')
    plt.plot(trucX, hisG, color = 'Green')
    plt.plot(trucX, hisB, color = 'Blue')
    plt.title('Biểu đồ Histogram')
    plt.xlabel('Giá trị mức xám')
    plt.ylabel('số điểm cùng giá trị RGB')
    plt.show()
    
#chương trình chính
#khai báo đường dẫn tới file hình
filehinh = r'bird_small.jpg'

#đọc ảnh dùng thư viện PIL
imgPIL = Image.open(filehinh)

#tính histogram
hisR, hisG, hisB = TinhHistogram(imgPIL)

#chuyển ảnh PIL sang OpenCV để hiện thị bằng thư viện CV2, không sử dụng được vì PIL là BGR còn open CV là RGB
#HinhRGB = np.array(imgPIL)

img = cv2.imread('bird_small.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Anh mau RGB', img)

#Hiển thị biểu đồ Histogram
VebieudoHistogram(hisR, hisG, hisB)

# bấm phím bất kì để đóng cửa sổ hiện thị hình
cv2.waitKey(0)

#giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình
cv2.destroyAllWindows()