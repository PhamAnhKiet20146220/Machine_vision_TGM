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
        Bitmap HinhGoc;
        public Form1()
        {
            InitializeComponent();
            // load hình từ file
            HinhGoc = new Bitmap(@"D:\Machine Vision\lena_color.jpg");

            //cho hiển thị lên pixBoxHinhGoc
            picBoxHinhGoc.Image = HinhGoc;

        }

        // sử dụng hàm chuyển ảnh sang mức xám bằng pp average
        public Bitmap ChuyenHinhRGBSangHinhXamAverage(Bitmap HinhGoc) //hàm mức xám theo pp Average
        {
            Bitmap HinhMucXam = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            for (int x = 0; x < HinhGoc.Width; x++) //tạo hàm for đi theo chiều ngang
                for (int y = 0; y < HinhGoc.Height; y++) // tạo hàm for đi theo chiều từ trên xuống
                {
                    //lấy điểm ảnh
                    Color pixel = HinhGoc.GetPixel(x, y);
                    byte R = pixel.R;
                    byte G = pixel.G;
                    byte B = pixel.B;

                    //tính giá trị mức xám cho điểm ảnh tại (x,y)
                    byte gray = (byte)((R + G + B) / 3); // ép kết quả về kiểu byte

                    // gán giá trị mức xám vừa tính vào hình mức xám
                    HinhMucXam.SetPixel(x, y, Color.FromArgb(gray, gray, gray));
                }
            return HinhMucXam;
        }

        // tạo hàm nhận dạng đường biên dùng phương pháp Sobel cho ảnh xám
        public Bitmap phandoantheoSobel(Bitmap HinhMucXam, byte Nguong)
        {
            // tạo 2 ma trận sobel theo x và y
            int[,] MatranX =
            {
                {-1, -2, -1 },
                { 0,  0,  0 },
                { 1,  2,  1 },
            };
            int[,] MatranY =
            {
                {-1, 0, 1 },
                {-2, 0, 2 },
                {-1, 0, 1 },
            };

            //tạo ảnh chứa để trả về
            Bitmap anhphandoan = new Bitmap(HinhMucXam.Width, HinhMucXam.Height);

            for (int x = 1; x < HinhMucXam.Width - 1; x++)
                for (int y = 1; y < HinhMucXam.Height - 1; y++)
                {
                    int Gx = 0, Gy = 0;

                    //tạo vòng lặp để nhân tích chập 2 ma trận cho cả x và y
                    for (int i = x - 1; i <= x + 1; i++)
                        for (int j = y - 1; j <= y + 1; j++)
                        {
                            Color color = HinhMucXam.GetPixel(i, j);

                            int gray = color.R;                           //trong ảnh xám thì cả 3 kênh đều như nhau

                            Gy += gray * MatranY[i - x + 1, j - y + 1];
                            

                        }

                    for (int a = x - 1; a <= x + 1; a++)
                        for (int b = y - 1; b <= y + 1; b++)
                        {
                            Color color1 = HinhMucXam.GetPixel(a, b);

                            int gray1 = color1.R;                           //trong ảnh xám thì cả 3 kênh đều như nhau

                            Gx += gray1 * MatranX[a - x + 1, b - y + 1];

                        }

                    int Mxy = Math.Abs(Gx) + Math.Abs(Gy);

                    if (Mxy < Nguong)
                        anhphandoan.SetPixel(x, y, Color.FromArgb(0, 0, 0));

                    else
                        anhphandoan.SetPixel(x, y, Color.FromArgb(255, 255, 255));
                }

            return anhphandoan;
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

        private void picBoxHinhGoc_Click(object sender, EventArgs e)
        {

        }

        private void hScrollBarNhiPhan_Scroll(object sender, ScrollEventArgs e)
        {
            // lấy giá trị ngưỡng từ giá trị của thanh cuộn
            // ép kiểu int của thanh cuộn về byte
            byte nguong = (byte)hScrollBarNhiPhan.Value;

            //cho hiển thị giá trị ngưỡng:
            lblNguong.Text = nguong.ToString();

            //gọi hàm tính sang hình xám
            Bitmap hinhxam = ChuyenHinhRGBSangHinhXamAverage(HinhGoc);

            //gọi hàm tính phân đoạn và cho hiển thị:
            pictBoxC.Image = phandoantheoSobel(hinhxam, nguong);
        }
    }
}
