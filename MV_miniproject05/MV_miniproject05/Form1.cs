using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZedGraph;

namespace MV_miniproject05
{
    public partial class Form1 : Form
    {   // chuyển biến Bitmap HinhGoc thành biến toàn cục
        Bitmap HinhGoc;
        public Form1()
        {
            InitializeComponent();
            //tạo bitmap để load hình
            HinhGoc = new Bitmap(@"D:\Machine Vision\bird_small.jpg");

            // Hiển thị hình gốc lên Picbox hình gốc
            picBoxHinhGoc.Image = HinhGoc;

            //hình màu mức xám theo pp Linear Luminace và cho hiển thị: 
            Bitmap HinhMucXam = ChuyenHinhRGBSangHinhXamLuminance(HinhGoc);
            picBoxHinhMucXam.Image = HinhMucXam;

            //hàm tính histogram
            double[] histogram = TinhHistogram(HinhMucXam);

            //chuyển đổi kiểu dữ liệu
            PointPairList points = ChuyenDoiHistogram(histogram);

            //vẽ biểu đồ và hiển thị
            zGHistogram.GraphPane = BieuDoHistogram(points);
            zGHistogram.Refresh();
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
                    byte gray = (byte)(0.2126 * R + 0.7152 * G + 0.0722 * B); // ép kết quả về kiểu byte

                    // gán giá trị mức xám vừa tính vào hình mức xám
                    HinhMucXam.SetPixel(x, y, Color.FromArgb(gray, gray, gray));
                }
            return HinhMucXam;
        }

        public double[] TinhHistogram(Bitmap Hinhmucxam)    //tính histogram của ảnh màu xám
        {
            // mỗi pixel mức xám có giá trị từ 0 - 255, do vậy ta sài 1 mảng có 256 phần tử dùng để chứa số đếm 
            // của các pixel có cùng mức xám trong ảnh. Kiểu double vì tổng số đếm lớn
            double[] histogram = new double[256];

            for (int x = 0; x< Hinhmucxam.Width; x++)
                for(int y = 0;y< Hinhmucxam.Height; y++)
                {
                    Color color = Hinhmucxam.GetPixel(x, y);
                    byte gray = color.R; //trong hình mức xám thì giá trị kênh R cũng giống G và B

                    //giá trị gray tính ra cũng chính là phần tử thứ gray trong mảng histogram
                    //tăng số đếm của phần tử thứ gray lên 1
                    histogram[gray]++;
                }
            return histogram;
        }

        PointPairList ChuyenDoiHistogram(double[] histogram)
        {
            //pointpairlist là kiểu dữ liệu của Zedgraph  để vẽ biểu đồ
            PointPairList points = new PointPairList();

            for (int i = 0;i< histogram.Length;i++)
            {
                //i tương ứng trục nằm ngang, từ 0 - 255
                //histogram[i] tương ứng trục đứng, là số pixels cùng mức xám
                points.Add(i, histogram[i]);
            }
            return points;
        }

        public GraphPane BieuDoHistogram(PointPairList histogram)
        {
            //GraphPane là đối tượng biểu đồ trong ZedGraph
            GraphPane gp = new GraphPane();

            gp.Title.Text = @"Biểu đồ Histogram";
            gp.Rect = new Rectangle(0, 0, 700, 500); //Khung chứa biểu đồ

            //Thiết lập trục ngang
            gp.XAxis.Title.Text = @"Giá trị mức xám của các điểm ảnh";
            gp.XAxis.Scale.Min = 0; //nhỏ nhất là 0
            gp.XAxis.Scale.Max = 255; //lớn nhất là 255
            gp.XAxis.Scale.MajorStep = 5; // mỗi bước chính là 5
            gp.XAxis.Scale.MinorStep = 1; //mỗi bước trong 1 bước chính là 1

            // tương tự cho trục đứng
            gp.YAxis.Title.Text = @"Số điểm ảnh có cùng mức xám";
            gp.YAxis.Scale.Min = 0; 
            gp.YAxis.Scale.Max = 15000; //số này phải lớn hơn kích thước ảnh (w x h) 
            gp.YAxis.Scale.MajorStep = 5; 
            gp.YAxis.Scale.MinorStep = 1;

            //Dùng biểu đồ dạng bar để biểu diễn histogram
            gp.AddBar("Histogram", histogram, Color.OrangeRed);

            return gp;
        }
            private void picBoxHinhGoc_Click(object sender, EventArgs e)
        {

        }
    }
}
