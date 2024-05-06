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

            // dùng hàm chuyển đổi RGB sang HSI
            List<Bitmap> HSI = ChuyenDoiRGBSangXYZ(HinhGoc);

            // sau đó cho hiển thị kết quả
            pictBoxH.Image = HSI[0];
            picBoxS.Image = HSI[1];
            picBoxI.Image = HSI[2];
            picBoxHSI.Image = HSI[3];
            
        }

        // Hàm chuyển đổi RGB sang XYZ
        public List<Bitmap> ChuyenDoiRGBSangXYZ(Bitmap HinhGoc)
        {
            // tạo mảng động list chứa các hình kết quả sau chuyển đổi
            List<Bitmap> HSI = new List<Bitmap>();

            // tạo 3 kênh màu để chứa hình của các kênh H-S-I
            Bitmap X = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Y = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Z = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            //hình HSI (kết hợp cả 3 kênh H-S-I)
            Bitmap XYZ = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            //tiến hành quét từng điểm ảnh có trong hình
            for (int x = 0; x < HinhGoc.Width; x++)
                for (int y = 0; y < HinhGoc.Height; y++)
                {
                    // lấy thông tin điểm ảnh tại vị trí x,y
                    Color pixel = HinhGoc.GetPixel(x, y);

                    // ở những bài trước dùng kiểu byte, tuy nhiên XYZ thì phải dùng double
                    // do lúc tính toán, các giá trị trả về kết quả đều là kiểu double (số thực)
                    double R = pixel.R;
                    double G = pixel.G;
                    double B = pixel.B;

                    // dựa vào công thức trong sách, ta có:
                    // Tạo mảng 3x3 theo công thức:
                    double[,] matrix = new double[3, 3];
                    matrix = new double[,]
                    {
                        {0.4124564, 0.3575761, 0.1804375},
                        {0.2126729, 0.7151522, 0.0721750},
                        {0.0193339, 0.1191920, 0.9503041}
                    };

                    // tạo mảng chứa 3 giá trị GRB
                    double[] rgb = new double[] { R, G, B };

                    // tạo mảng chứa kết quả sau khi nhân 2 ma trận
                    double[] result = new double[3];

                    // tạo vòng lặp để nhân 2 ma trận
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            result[i] += matrix[i, j] * rgb[j];
                        }
                    }

                    //Tính X-Y-Z bằng thủ công
                    //double X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B);
                    //double Y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B);
                    //double Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B);



                    // cho hiển thị các kênh XYZ tại cái picbox
                    // ép kiểu các giá trị XYZ về kiểu byte 
                    X.SetPixel(x, y, Color.FromArgb((byte)result[0], (byte)result[0], (byte)result[0]));
                    Y.SetPixel(x, y, Color.FromArgb((byte)result[1], (byte)result[1], (byte)result[1]));
                    Z.SetPixel(x,y, Color.FromArgb((byte)result[2], (byte)result[2], (byte)result[2]));

                    XYZ.SetPixel(x,y,Color.FromArgb((byte)result[0], (byte)result[1], (byte)result[2])); 

                }
            // đã khai báo 1 mảng động ở trên để chứa kết quả các hình, trả về sau khi chuyển đổi
            HSI.Add(X);
            HSI.Add(Y);
            HSI.Add(Z);
            HSI.Add(XYZ);

            // trả mảng sau khi chuyển đổi cho hàm
            return HSI;

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

        private void pictBoxC_Click(object sender, EventArgs e)
        {

        }

        private void lblIntensity_Click(object sender, EventArgs e)
        {

        }

        private void lblHue_Click(object sender, EventArgs e)
        {

        }
    }
}
