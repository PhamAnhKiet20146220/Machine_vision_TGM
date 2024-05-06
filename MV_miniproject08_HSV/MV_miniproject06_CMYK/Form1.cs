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
            List<Bitmap> HSI = ChuyenDoiRGBSangHSV(HinhGoc);

            // sau đó cho hiển thị kết quả
            pictBoxH.Image = HSI[0];
            picBoxS.Image = HSI[1];
            picBoxI.Image = HSI[2];
            picBoxHSI.Image = HSI[3];
            
        }

        // Hàm chuyển đổi RGB sang HSV
        public List<Bitmap> ChuyenDoiRGBSangHSV(Bitmap HinhGoc)
        {
            // tạo mảng động list chứa các hình kết quả sau chuyển đổi
            List<Bitmap> HSI = new List<Bitmap>();

            // tạo 3 kênh màu để chứa hình của các kênh H-S-I
            Bitmap Hue = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Saturation = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Intensity = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            //hình HSI (kết hợp cả 3 kênh H-S-I)
            Bitmap HSIImg = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            //tiến hành quét từng điểm ảnh có trong hình
            for (int x = 0; x < HinhGoc.Width; x++)
                for (int y = 0; y < HinhGoc.Height; y++)
                {
                    // lấy thông tin điểm ảnh tại vị trí x,y
                    Color pixel = HinhGoc.GetPixel(x, y);

                    // ở những bài trước dùng kiểu byte, tuy nhiên HSI thì phải dùng double
                    // do lúc tính toán, các giá trị H-S-I trả về kết quả đều là kiểu double (số thực)
                    double R = pixel.R;
                    double G = pixel.G;
                    double B = pixel.B;

                    // dựa vào công thức trong sách, ta có:
                    // phần tử số của công thức:
                    double t1 = ((R - G) + (R - B)) / 2;

                    //phần mẫu số của công thức tính góc theta
                    //double t2 = (R - G) * (R - G) + Math.Sqrt((R - B) * (G - B));
                    
                    double t2 = Math.Sqrt((R - G) * (R - G) + (R - B) * (G - B));

                    //kết quả hàm tính Acos trong C# là radian
                    double theta = Math.Acos(t1 / t2);
                    // công thức tính giá trị Hue
                    double H = 0;

                    // nếu blue <= green thì Hue = theta
                    if (B <= G)
                        H = theta;
                    else // ngược lại blue > green thì Hue = :
                        H = 2*Math.PI - theta; // do theta tính đc ở trên là radian nên dùng Pi

                    // chuyển đổi từ radian sang độ
                    H = H*180/Math.PI;

                    // công thức tính giá trị kênh Saturation
                    double S = 1 - 3*Math.Min(R, Math.Min(G,B)) / (R + G + B);

                    // Do giá trị tính ra của S trong khoảng [0,1]
                    // để bitmap có thể hiển thị được thì phải convert S sang [0,255]
                    // S = S * 255

                    // công thức tính giá trị Value
                    double V = Math.Max(R, Math.Max(G, B)) ;

                    // cho hiển thị các kênh HSI tại cái picbox
                    // ép kiểu các giá trị H - S - V về kiểu byte 
                    Hue.SetPixel(x, y, Color.FromArgb((byte)H, (byte)H, (byte)H));
                    
                    // giá trị S chúng ta chỉ normalized lúc hiển thị
                    Saturation.SetPixel(x, y, Color.FromArgb((byte)(S * 255), (byte)(S * 255), (byte)(S * 255)));
                    Intensity.SetPixel(x,y, Color.FromArgb((byte)V, (byte)V, (byte)V));

                    // giá trị gốc của S dùng để tạo hình HSV. Hình HSI để quan sát, còn tính tính vẫn dùng các kênh riêng lẽ
                    // với tửng mục đính ảnh khác nhau
                    HSIImg.SetPixel(x,y,Color.FromArgb((byte)H, (byte)(S*255), (byte)V)); // muốn như ảnh trên gg.doc thì S*255

                }
            // đã khai báo 1 mảng động ở trên để chứa kết quả các hình, trả về sau khi chuyển đổi
            HSI.Add(Hue);
            HSI.Add(Saturation);
            HSI.Add(Intensity);
            HSI.Add(HSIImg);

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
    }
}
