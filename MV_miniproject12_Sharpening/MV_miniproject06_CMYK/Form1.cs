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
            Bitmap sharpen = Sharpening(HinhGoc);


            //cho hiển thị kết quả đã làm mượt lên picbox
            pictBoxC.Image = sharpen;

        }

        public Bitmap Sharpening(Bitmap HinhGoc)
        {
            //tạo 1 bitmap chứ ảnh đã làm mượt 3x3
            Bitmap sharpness = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // Tạo mảng 3x3 theo công thức:
            int[,] matrix = new int[3, 3];
            matrix = new int[,]
            {
                        { 0, -1, 0 }, 
                        { -1, 4, -1}, 
                        { 0, -1, 0 }
            };

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
                            Rs += R * matrix[i - x + 1, j - y + 1];
                            Gs += G * matrix[i - x + 1, j - y + 1];
                            Bs += B * matrix[i - x + 1, j - y + 1];
                        }
                    Color mau = HinhGoc.GetPixel(x, y);

                    byte Rm = mau.R;
                    byte Gm = mau.G;
                    byte Bm = mau.B;

                    int tongR = Rm + Rs;
                    int tongG = Gm + Gs;
                    int tongB = Bm + Bs;

                    if (tongR >= 255)
                        tongR = 255;
                    if (tongG >= 255)
                        tongG = 255;
                    if (tongB >= 255)
                        tongB = 255;

                    if (tongR <= 0)
                        tongR = 0;
                    if (tongG <= 0)
                        tongG = 0;
                    if (tongB <= 0)
                        tongB = 0;

                    //set điểm ảnh đã làm nét vào ảnh bitmap

                    sharpness.SetPixel(x,y, Color.FromArgb((byte)tongR, (byte)tongG, (byte)tongB));
                }
            // trả ảnh đã làm nét
            return sharpness;
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
    }
}
