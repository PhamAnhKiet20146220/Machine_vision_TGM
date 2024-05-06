using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Grayscale_Lightness
{
    public partial class Form1 : Form
    {   // chuyển biến Bitmap HinhGoc thành biến toàn cục
        Bitmap HinhGoc;
        public Form1()
        {
            InitializeComponent();
            //tạo bitmap để load hình
            HinhGoc = new Bitmap(@"D:\Machine Vision\lena_color.jpg");

            // Hiển thị hình gốc lên Picbox hình gốc
            picBoxHinhGoc.Image = HinhGoc;

            //hình màu mức xám theo pp Lightness và cho hiển thị: 
            picBoxHinhxamLightness.Image = ChuyenHinhRGBSangHinhXamLightness(HinhGoc);

            //hình màu mức xám theo pp Average và cho hiển thị: 
            picBoxHinhXamAverage.Image = ChuyenHinhRGBSangHinhXamAverage(HinhGoc);

            //hình màu mức xám theo pp Linear Luminace và cho hiển thị: 
            picBoxHinhXamLuminance.Image = ChuyenHinhRGBSangHinhXamLuminance(HinhGoc);

            //tính hình nhị phân và cho hiển thị:
            picBoxHinhNhiPhan.Image = ChuyenHinhRGBSangnhiphan(HinhGoc, 100); //giá trị ngưỡng là 100

        }

        public Bitmap ChuyenHinhRGBSangHinhXamLightness(Bitmap HinhGoc) //hàm mức xám theo pp lightness
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
                    byte max = Math.Max(R, Math.Max(G, B)); //so sánh giá trị lớn nhất trong 3 giá trị
                    byte mix = Math.Min(R, Math.Min(G, B)); //so sánh giá trị nhỏ nhất trong 3 giá trị
                    byte gray = (byte)((max+mix)/2); // ép kết quả về kiểu byte

                    // gán giá trị mức xám vừa tính vào hình mức xám
                    HinhMucXam.SetPixel(x, y, Color.FromArgb(gray, gray, gray));
                }
            return HinhMucXam;
        }

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

        public Bitmap ChuyenHinhRGBSangHinhXamLuminance(Bitmap HinhGoc) //hàm mức xám theo pp độ sáng tuyến tính linear Luminance
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
                    byte gray = (byte)(0.2126*R + 0.7152*G + 0.0722*B); // ép kết quả về kiểu byte

                    // gán giá trị mức xám vừa tính vào hình mức xám
                    HinhMucXam.SetPixel(x, y, Color.FromArgb(gray, gray, gray));
                }
            return HinhMucXam;
        }
        public Bitmap ChuyenHinhRGBSangnhiphan(Bitmap HinhGoc, byte nguong) //hàm chuyển RGB sang nhị phân
        {
            Bitmap HinhNhiPhan = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            for (int x = 0; x < HinhGoc.Width; x++) //tạo hàm for đi theo chiều ngang
                for (int y = 0; y < HinhGoc.Height; y++) // tạo hàm for đi theo chiều từ trên xuống
                {
                    //lấy điểm ảnh
                    Color pixel = HinhGoc.GetPixel(x, y);
                    byte R = pixel.R;
                    byte G = pixel.G;
                    byte B = pixel.B;

                    //tính giá trị mức xám cho điểm ảnh tại (x,y)
                    byte nhiphan = (byte)(0.2126 * R + 0.7152 * G + 0.0722 * B); // ép kết quả về kiểu byte

                    //phân loại điểm ảnh sang nhị phân dựa vào giá trị của ngưỡng
                    if (nhiphan < nguong)
                        nhiphan = 0;
                    else
                        nhiphan = 255;

                    // gán giá trị nhị phân vừa tính vào hình nhị phân
                    HinhNhiPhan.SetPixel(x, y, Color.FromArgb(nhiphan, nhiphan, nhiphan));
                }
            return HinhNhiPhan;

         
        }
        private void lblHinhGoc_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void lblHinhXamAvergare_Click(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click_1(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
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

            //gọi hàm tính nhị phân và cho hiển thị:
            picBoxHinhNhiPhan.Image = ChuyenHinhRGBSangnhiphan(hinhxam, nguong);

        }
    }
}