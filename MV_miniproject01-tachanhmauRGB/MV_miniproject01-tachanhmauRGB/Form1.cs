using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MV_miniproject01_tachanhmauRGB
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            // Biến chứa đường dẫn hình gốc:
            // @ trước chuỗi để biết là chuỗi Unicode có kí tự đặt biệt
            string filehinh = @"D:\Machine Vision\lena_color.jpg";

            // tạo biến chứa hình bitmap được load từ file hình:
            Bitmap hinhgoc = new Bitmap(filehinh);

            // Hiển thị hình gốc tại picBox hình gốc:
            picBox_Hinhgoc.Image = hinhgoc;
            
            //Khai báo 3 hình bitmap để chứa 3 hình kênh màu R G B:
            Bitmap red = new Bitmap(hinhgoc.Width, hinhgoc.Height);
            Bitmap green = new Bitmap(hinhgoc.Width, hinhgoc.Height);
            Bitmap blue = new Bitmap(hinhgoc.Width, hinhgoc.Height);

            // mỗi hình là một ma trận 2 chiều nên dùng hàm for để đọc hết các điểm ảnh (pixel) có trong hình
            for (int x = 0; x < hinhgoc.Width; x++)
                for (int y = 0; y < hinhgoc.Height; y++)
                {
                    // đọc giá trị pixel tại điểm ảnh có vị trí (x,y):
                    Color pixel = hinhgoc.GetPixel(x, y);
                    // mỗi  pixel chứa 4 thông tin gồm giá trị màu RGB và độ trong suốt (A) tại điểm xét:
                    byte R = pixel.R; //giá trị kênh red
                    byte G = pixel.G; //giá trị kênh green
                    byte B = pixel.B; //giá trị kênh blue
                    byte A = pixel.A; //giá trị độ trong suốt

                    // set giá trị pixel đọc được cho các hình chứa các màu tương ứng RGB.
                    red.SetPixel(x, y, Color.FromArgb(A, R, 0, 0));
                    green.SetPixel(x, y, Color.FromArgb(A, 0, G, 0));
                    blue.SetPixel(x, y, Color.FromArgb(A, 0, 0, B));
                }
            // hiển thị hình ảnh tại các picbox tương ứng
            picBox_RED.Image = red;
            picBox_GREEN.Image = green;
            picBox_BLUE.Image = blue;
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }
    }
}
