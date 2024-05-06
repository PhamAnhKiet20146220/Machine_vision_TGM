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


        // tạo hàm nhận dạng đường biên dùng phương pháp Sobel cho ảnh RGB
        public Bitmap phandoantheoSobel(Bitmap HinhGoc, byte Nguong)
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
            Bitmap anhphandoan = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            for (int x = 1; x < HinhGoc.Width - 1; x++)
                for (int y = 1; y < HinhGoc.Height - 1; y++)
                {
                    int GxR = 0, GyR = 0;
                    int GxG = 0, GyG = 0;
                    int GxB = 0, GyB = 0;

                    //tạo vòng lặp để nhân tích chập 2 ma trận cho cả x và y
                    for (int i = x - 1; i <= x + 1; i++)
                        for (int j = y - 1; j <= y + 1; j++)
                        {
                            Color color = HinhGoc.GetPixel(i, j);

                            int R = color.R;                           
                            int G = color.G;
                            int B = color.B;

                            GxR += R * MatranX[i - x + 1, j - y + 1];
                            GyR += R * MatranY[i - x + 1, j - y + 1];

                            GxG += G * MatranX[i - x + 1, j - y + 1];
                            GyG += G * MatranY[i - x + 1, j - y + 1];

                            GxB += B * MatranX[i - x + 1, j - y + 1];
                            GyB += B * MatranY[i - x + 1, j - y + 1];


                        }
                    double Gxx = Math.Abs(GxR)* Math.Abs(GxR) + Math.Abs(GxG)* Math.Abs(GxG) + Math.Abs(GxB)* Math.Abs(GxB);
                    double Gyy = Math.Abs(GyR)* Math.Abs(GyR) + Math.Abs(GyG)* Math.Abs(GyG) + Math.Abs(GyB)* Math.Abs(GyB);
                    double Gxy = GxR*GyR + GxG * GyG + GxB * GyB;

                    double theta = 0.5 * Math.Atan2 ( (2 * Gxy) , ( Gxx - Gyy ) );
                    //double theta = Math.Atan2(Gyy, Gxx);

                    if (Gxx > 0)
                        theta = Math.Atan(Gyy / Gxx);

                    if (Gxx < 0 && Gyy >= 0)
                        theta = Math.Atan(Gyy / Gxx) + Math.PI;

                    if (Gxx < 0 && Gyy < 0)
                        theta = Math.Atan(Gyy / Gxx) - Math.PI;

                    if (Gxx == 0 && Gyy > 0)
                        theta = Math.PI;

                    if (Gxx == 0 && Gyy < 0)
                        theta = -Math.PI;

                    double F0 = Math.Sqrt( 0.5 * ((Gxx + Gyy) + (Gxx - Gyy ) * Math.Cos(2 * theta) + 2*Gxy*Math.Sin(2*theta) ));

                    if (F0 <= Nguong)
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

            //gọi hàm tính phân đoạn và cho hiển thị:
            pictBoxC.Image = phandoantheoSobel(HinhGoc, nguong);

        }
    }
}
