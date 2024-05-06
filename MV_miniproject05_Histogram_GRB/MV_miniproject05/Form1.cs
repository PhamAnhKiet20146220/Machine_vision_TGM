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

            //hàm tính histogram
            double[,] histogram = TinhHistogram(HinhGoc);

            //chuyển đổi kiểu dữ liệu
            List<PointPairList> points = ChuyenDoiHistogram(histogram);

            //vẽ biểu đồ và hiển thị
            zGHistogram.GraphPane = BieuDoHistogram(points);
            zGHistogram.Refresh();

        }

        public double[,] TinhHistogram(Bitmap bpm)    //tính histogram của ảnh màu RGB
        {   
            //dùng mảng 2 chiều để chứa thông tin histogram cho các kênh RGB
            double[,] histogram = new double[3, 256];     //3 là số kênh màu cần lưu, 256 là vị trí tương ứng giá trị màu tự 0-255

            for (int x = 0; x < bpm.Width; x++)
                for (int y = 0; y < bpm.Height; y++)
                {
                    Color color = bpm.GetPixel(x, y);
                    byte R = color.R;
                    byte G = color.G;
                    byte B = color.B;

                    histogram[0, R]++; //histogram của kênh màu Red
                    histogram[1, G]++; //histogram của kênh màu Green
                    histogram[2, B]++; //histogram của kênh màu Blue

                }
            return histogram; // trả mảng 2 chiều chứa thông tin histogram của RGB
        }

        List <PointPairList> ChuyenDoiHistogram(double[,] histogram)
        {
            //dùng mảng không cần khai báo trước số lượng phần tử để chứa các chuyển đổi
            List <PointPairList> points = new List<PointPairList> ();

            PointPairList redpoint = new PointPairList (); //chuyển đổi histogram kênh red
            PointPairList greenpoint = new PointPairList(); //chuyển đổi histogram kênh green
            PointPairList bluepoint = new PointPairList(); //chuyển đổi histogram kênh blue

            for (int i = 0; i < 256; i++)
            {
                //i tương ứng trục nằm ngang, từ 0 - 255
                //histogram[i] tương ứng trục đứng, là số pixels cùng mức xám
                redpoint.Add(i, histogram[0, i]); //chuyển đổi cho kênh R
                greenpoint.Add(i, histogram[1, i]); //chuyển đổi cho kênh G
                bluepoint.Add(i, histogram[2, i]); //chuyển đổi cho kênh B
            }

            //hết vòng for, chuyển đổi thông tin histogram RGB thành công
            //add các kênh vào mảng points để trả về cho hàm
            points.Add (redpoint);
            points.Add (greenpoint);
            points.Add (bluepoint);

            return points;
        }

        public GraphPane BieuDoHistogram(List <PointPairList> histogram)
        {
            //GraphPane là đối tượng biểu đồ trong ZedGraph
            GraphPane gp = new GraphPane();

            gp.Title.Text = @"Biểu đồ Histogram";
            gp.Rect = new Rectangle(0, 0, 700, 500); //Khung chứa biểu đồ

            //Thiết lập trục ngang
            gp.XAxis.Title.Text = @"Giá trị màu của các điểm ảnh";
            gp.XAxis.Scale.Min = 0; //nhỏ nhất là 0
            gp.XAxis.Scale.Max = 255; //lớn nhất là 255
            gp.XAxis.Scale.MajorStep = 5; // mỗi bước chính là 5
            gp.XAxis.Scale.MinorStep = 1; //mỗi bước trong 1 bước chính là 1

            // tương tự cho trục đứng
            gp.YAxis.Title.Text = @"Số điểm ảnh có cùng giá trị màu";
            gp.YAxis.Scale.Min = 0;
            gp.YAxis.Scale.Max = 15000; //số này phải lớn hơn kích thước ảnh (w x h) 
            gp.YAxis.Scale.MajorStep = 5;
            gp.YAxis.Scale.MinorStep = 1;

            //Dùng biểu đồ dạng bar để biểu diễn histogram
            gp.AddBar("Histogram's Red", histogram[0], Color.Red);
            gp.AddBar("Histogram's Green", histogram[1], Color.Green);
            gp.AddBar("Histogram's Blue", histogram[2], Color.Blue);

            return gp;
        }

        private void picBoxHinhGoc_Click(object sender, EventArgs e)
        {

        }
    }
}
