using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MV_miniproject06_CMYK
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            // load hình từ file
            Bitmap HinhGoc = new Bitmap(@"D:\Machine Vision\lena_color.jpg");

            //cho hiển thị lên pixBoxHinhGoc
            picBoxHinhGoc.Image = HinhGoc;

            //sử dụng hàm làm mượt cho ảnh gốc lena
            //Bitmap SmoothedImage3x3 = ColorImageSmoothing3x3(HinhGoc);
            //Bitmap SmoothedImage5x5 = ColorImageSmoothing5x5(HinhGoc);
            //Bitmap SmoothedImage7x7 = ColorImageSmoothing7x7(HinhGoc);
            //Bitmap SmoothedImage9x9 = ColorImageSmoothing9x9(HinhGoc);

            Bitmap SmoothedImage3x3 = chuyendoi(HinhGoc, 3);
            Bitmap SmoothedImage5x5 = chuyendoi(HinhGoc, 5);
            Bitmap SmoothedImage7x7 = chuyendoi(HinhGoc, 7);
            Bitmap SmoothedImage9x9 = chuyendoi(HinhGoc, 9);

            //cho hiển thị kết quả đã làm mượt lên picbox
            pictBoxC.Image = SmoothedImage3x3;
            picBoxM.Image = SmoothedImage5x5;
            picBoxY.Image = SmoothedImage7x7;
            picBoxB.Image = SmoothedImage9x9;
        }

        public Bitmap chuyendoi(Bitmap HinhGoc, int K) //k = 3,5,7,9
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap SmothedImage = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            int giatriK = K / 2; 

            // quét điểm ảnh RGB gốc, với mặt nạ 3x3 thì có thể bỏ qua các viền ngoài của ảnh
            // do v chỉ quét từ x = 1 đến x = width - 1 và y = 1 đến y = heigh -1
            for (int x = giatriK; x < HinhGoc.Width - giatriK; x++)
                for (int y = giatriK; y < HinhGoc.Height - giatriK; y++)
                {
                    // các biến chứa giá trị cộng dồn của các điểm ảnh nên được khai báo là int
                    // để chứa giá trị cộng dồn của các pixel
                    int Rs = 0, Gs = 0, Bs = 0;

                    //tiến hành quét các điểm trong mặt nạ
                    for (int i = x - giatriK; i <= x + giatriK; i++)
                        for (int j = y - giatriK; j <= y + giatriK; j++)
                        {
                            //lấy thông tin màu RGB tại điểm ảnh trong mặt nạ, tại vị trí (i,j)
                            Color color = HinhGoc.GetPixel(i, j);
                            byte R = color.R;
                            byte G = color.G;
                            byte B = color.B;

                            //cộng dồn tất cả điểm ảnh đó cho mỗi kênh R-G-B tương ứng
                            Rs += R;
                            Gs += G;
                            Bs += B;
                        }
                    //kết thúc quét và cộng dồn điểm ảnh trong mặt nạ, dùng công thức 6.6-2 cho từng kênh RGB
                    int matna = K * K;
                    Rs = (int)(Rs / matna);
                    Gs = (int)(Gs / matna);
                    Bs = (int)(Bs / matna);

                    SmothedImage.SetPixel(x, y, Color.FromArgb(Rs, Gs, Bs));
                }
            return SmothedImage;
        }

        public Bitmap ColorImageSmoothing3x3(Bitmap HinhGoc)
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap SmothedImage3x3 = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // quét điểm ảnh RGB gốc, với mặt nạ 3x3 thì có thể bỏ qua các viền ngoài của ảnh
            // do v chỉ quét từ x = 1 đến x = width - 1 và y = 1 đến y = heigh -1
            for (int x = 1;  x < HinhGoc.Width - 1; x++)
                for (int y = 1; y < HinhGoc.Height - 1; y++)
                {
                    // các biến chứa giá trị cộng dồn của các điểm ảnh nên được khai báo là int
                    // để chứa giá trị cộng dồn của các pixel
                    int Rs = 0, Gs = 0, Bs = 0;

                    //tiến hành quét các điểm trong mặt nạ
                    for (int i = x - 1; i <= x + 1; i++)
                        for (int j = y - 1; j <= y + 1; j++)
                        {
                            //lấy thông tin màu RGB tại điểm ảnh trong mặt nạ, tại vị trí (i,j)
                            Color color = HinhGoc.GetPixel(i, j);
                            byte R = color.R;
                            byte G = color.G;
                            byte B = color.B;

                            //cộng dồn tất cả điểm ảnh đó cho mỗi kênh R-G-B tương ứng
                            Rs += R;
                            Gs += G;
                            Bs += B;
                        }
                    //kết thúc quét và cộng dồn điểm ảnh trong mặt nạ, dùng công thức 6.6-2 cho từng kênh RGB
                    byte K = 3 * 3;
                    Rs = (int)(Rs / K);
                    Gs = (int)(Gs / K);
                    Bs = (int)(Bs / K);

                    //set điểm ảnh đã làm mượt ( mờ ) vào ảnh bitmap
                    SmothedImage3x3.SetPixel(x,y, Color.FromArgb(Rs, Gs, Bs));
                }
            // trả ảnh đã làm mượt
            return SmothedImage3x3;
        }

        public Bitmap ColorImageSmoothing5x5(Bitmap HinhGoc)
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap SmothedImage5x5 = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // quét điểm ảnh RGB gốc, với mặt nạ 3x3 thì có thể bỏ qua các viền ngoài của ảnh
            // do v chỉ quét từ x = 1 đến x = width - 1 và y = 1 đến y = heigh -1
            for (int x = 2; x < HinhGoc.Width - 2; x++)
                for (int y = 2; y < HinhGoc.Height - 2; y++)
                {
                    // các biến chứa giá trị cộng dồn của các điểm ảnh nên được khai báo là int
                    // để chứa giá trị cộng dồn của các pixel
                    int Rs = 0, Gs = 0, Bs = 0;

                    //tiến hành quét các điểm trong mặt nạ
                    for (int i = x - 2; i <= x + 2; i++)
                        for (int j = y - 2; j <= y + 2; j++)
                        {
                            //lấy thông tin màu RGB tại điểm ảnh trong mặt nạ, tại vị trí (i,j)
                            Color color = HinhGoc.GetPixel(i, j);
                            byte R = color.R;
                            byte G = color.G;
                            byte B = color.B;

                            //cộng dồn tất cả điểm ảnh đó cho mỗi kênh R-G-B tương ứng
                            Rs += R;
                            Gs += G;
                            Bs += B;
                        }
                    //kết thúc quét và cộng dồn điểm ảnh trong mặt nạ, dùng công thức 6.6-2 cho từng kênh RGB
                    byte K = 5 * 5;
                    Rs = (int)(Rs / K);
                    Gs = (int)(Gs / K);
                    Bs = (int)(Bs / K);

                    //set điểm ảnh đã làm mượt ( mờ ) vào ảnh bitmap
                    SmothedImage5x5.SetPixel(x, y, Color.FromArgb(Rs, Gs, Bs));
                }
            // trả ảnh đã làm mượt
            return SmothedImage5x5;
        }

        public Bitmap ColorImageSmoothing7x7(Bitmap HinhGoc)
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap SmothedImage7x7 = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // quét điểm ảnh RGB gốc, với mặt nạ 3x3 thì có thể bỏ qua các viền ngoài của ảnh
            // do v chỉ quét từ x = 1 đến x = width - 1 và y = 1 đến y = heigh -1
            for (int x = 3; x < HinhGoc.Width - 3; x++)
                for (int y = 3; y < HinhGoc.Height - 3; y++)
                {
                    // các biến chứa giá trị cộng dồn của các điểm ảnh nên được khai báo là int
                    // để chứa giá trị cộng dồn của các pixel
                    int Rs = 0, Gs = 0, Bs = 0;

                    //tiến hành quét các điểm trong mặt nạ
                    for (int i = x - 3; i <= x + 3; i++)
                        for (int j = y - 3; j <= y + 3; j++)
                        {
                            //lấy thông tin màu RGB tại điểm ảnh trong mặt nạ, tại vị trí (i,j)
                            Color color = HinhGoc.GetPixel(i, j);
                            byte R = color.R;
                            byte G = color.G;
                            byte B = color.B;

                            //cộng dồn tất cả điểm ảnh đó cho mỗi kênh R-G-B tương ứng
                            Rs += R;
                            Gs += G;
                            Bs += B;
                        }
                    //kết thúc quét và cộng dồn điểm ảnh trong mặt nạ, dùng công thức 6.6-2 cho từng kênh RGB
                    byte K = 7 * 7;
                    Rs = (int)(Rs / K);
                    Gs = (int)(Gs / K);
                    Bs = (int)(Bs / K);

                    //set điểm ảnh đã làm mượt ( mờ ) vào ảnh bitmap
                    SmothedImage7x7.SetPixel(x, y, Color.FromArgb(Rs, Gs, Bs));
                }
            // trả ảnh đã làm mượt
            return SmothedImage7x7;
        }

        public Bitmap ColorImageSmoothing9x9(Bitmap HinhGoc)
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap SmothedImage9x9 = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // quét điểm ảnh RGB gốc, với mặt nạ 3x3 thì có thể bỏ qua các viền ngoài của ảnh
            // do v chỉ quét từ x = 1 đến x = width - 1 và y = 1 đến y = heigh -1
            for (int x = 4; x < HinhGoc.Width - 4; x++)
                for (int y = 4; y < HinhGoc.Height - 4; y++)
                {
                    // các biến chứa giá trị cộng dồn của các điểm ảnh nên được khai báo là int
                    // để chứa giá trị cộng dồn của các pixel
                    int Rs = 0, Gs = 0, Bs = 0;

                    //tiến hành quét các điểm trong mặt nạ
                    for (int i = x - 4; i <= x + 4; i++)
                        for (int j = y - 4; j <= y + 4; j++)
                        {
                            //lấy thông tin màu RGB tại điểm ảnh trong mặt nạ, tại vị trí (i,j)
                            Color color = HinhGoc.GetPixel(i, j);
                            byte R = color.R;
                            byte G = color.G;
                            byte B = color.B;

                            //cộng dồn tất cả điểm ảnh đó cho mỗi kênh R-G-B tương ứng
                            Rs += R;
                            Gs += G;
                            Bs += B;
                        }
                    //kết thúc quét và cộng dồn điểm ảnh trong mặt nạ, dùng công thức 6.6-2 cho từng kênh RGB
                    byte K = 9 * 9;
                    Rs = (int)(Rs / K);
                    Gs = (int)(Gs / K);
                    Bs = (int)(Bs / K);

                    //set điểm ảnh đã làm mượt ( mờ ) vào ảnh bitmap
                    SmothedImage9x9.SetPixel(x, y, Color.FromArgb(Rs, Gs, Bs));
                }
            // trả ảnh đã làm mượt
            return SmothedImage9x9;
        }
        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void picBoxY_Click(object sender, EventArgs e)
        {

        }

        private void lblCyan_Click(object sender, EventArgs e)
        {

        }
    }
}
