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

        private void btnPhanDoan_Click(object sender, EventArgs e)
        {
            //chuyển đổi sang uint cho các số lấy được từ text box
            int nguong = Convert.ToUInt16(textBox5.Text);

            int x1 = Convert.ToUInt16(textBox1.Text);
            int y1 = Convert.ToUInt16(textBox2.Text);

            int x2 = Convert.ToUInt16(textBox3.Text);
            int y2 = Convert.ToUInt16(textBox4.Text);

            double R, G, B;     // tạo 3 biến chưa giá trị trung bình RGB

            int RED = 0, GREEN = 0, BLUE = 0;       


            //Tính tổng màu RGB từ x1 dến x2 và y1 đến y2
            for (int x = x1; x <= x2; x++)
                for (int y = y1; y <= y2; y++)
                {
                    Color pixel = HinhGoc.GetPixel(x, y);
                    RED += pixel.R;             //cộng tất cả giá trị R trong vùng
                    GREEN += pixel.G;
                    BLUE += pixel.B;

                }

            // tiến hành chia cho kích thước vùng màu đã chọn để tính trung bình cộng
            int kichthuoc = Math.Abs(x2 - x1) * Math.Abs(y2 - y1);      //tính kích thước

            R = RED / kichthuoc;
            G = GREEN / kichthuoc;
            B = BLUE / kichthuoc;

            
            //Tạo 1 ảnh bitmap chứa hình sau khi phân đoạn
            Bitmap phandoananh = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // cần xác định cả hình nên cho Z là từ 0 - 512
            for (int x = 0; x < HinhGoc.Width; x++)
                for (int y = 0; y < HinhGoc.Height; y++)
                {
                    Color pixel = HinhGoc.GetPixel(x, y);
                    byte ZR = pixel.R;
                    byte ZG = pixel.G;
                    byte ZB = pixel.B;

                    //tính D theo công thức D(z, a) = sqrt((zR - aR)^2 + (zG - aG)^2 + (zB - aB)^2)
                    double D = Math.Sqrt(Math.Pow(ZR - R, 2) + Math.Pow(ZG - G, 2) + Math.Pow(ZB - B, 2));

                    //So sánh với GT ngưỡng D0 để xác định xem điểm Z(x,y) đang xét lớn hay nhỏ hơn
                    if ((int)D <= nguong)
                        phandoananh.SetPixel(x, y, Color.FromArgb(255, 255, 255));          //màu trắng
                    else
                        phandoananh.SetPixel(x, y, Color.FromArgb(ZR, ZG, ZB));             //màu gốc

                }
            //Hiển thị ảnh sau khi phân đoạn xong phân đoạn
            pictBoxC.Image = phandoananh;
        }
    }
}
