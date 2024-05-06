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

            //sử dụng hàm chuyển đổi RGB sang CMYK
            List<Bitmap> CMYK = ChuyenDoiRGBSangCMYK(HinhGoc);

            //hàm trên trả về 4 màu tương ứng thứ tự từ 0 - 3 là C-M-Y-K
            //Hiển thị các kênh màu CMYK được chuyển đổi từ RGB
            pictBoxC.Image = CMYK[0];       //kênh màu Cyan
            picBoxM.Image = CMYK[1];        //kênh màu Magenta
            picBoxY.Image = CMYK[2];        //kênh màu Yellow
            picBoxB.Image = CMYK[3];        //Kênh màu Black
        }

        public List<Bitmap> ChuyenDoiRGBSangCMYK(Bitmap HinhGoc)
        {
            //không cần dùng công thức, cụ thể chuyển đổi như sau:
            // Cyan (xanh dương) là sự kết hợp giữa Green và Blue, nên set kênh Red =0
            // Magenta (tím) là sự kết hợp giữa Red và Blue, nên set kênh Green =0
            // Yellow (vàng) là sự kết hợp giữa Red và Green, nên set kênh Blue =0
            // Black (Đen) là MIN (R,G,B)

            //tạo 1 list để chứa 4 kênh ảnh C-M-Y-K. Trong C#.net, list là 1 mảng nhưng không phải khai báo kích thước trước
            List<Bitmap> CMYK = new List<Bitmap>();

            // tạo 4 hình bitmap có kích thước bằng với kích thước hình gốc
            // để việc tính toán chuyển đổi kênh màu được thực hiện đúng với từng pixel
            Bitmap Cyan = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Magenta = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Yellow = new Bitmap(HinhGoc.Width, HinhGoc.Height);
            Bitmap Black = new Bitmap(HinhGoc.Width, HinhGoc.Height);

            // tạo 2 hàm for để quét ảnh, quét từng cột và từ trái qua phải
            for (int x = 0; x < HinhGoc.Width; x++)
                for (int y = 0; y < HinhGoc.Height; y++)
                {
                    //tiến hành lấy điểm ảnh, tại mỗi pixel, lấy thông tin RGB tại đó
                    Color pixel = HinhGoc.GetPixel(x, y);
                    byte R = pixel.R;       //mỗi kênh màu là 8bits = 1 byte nên dùng kiểu dữ liệu byte để chứa thông tin
                    byte G = pixel.G;
                    byte B = pixel.B;

                    //tiến hành trộn RGB để ra CMYK

                    //Màu Cyan (xanh dương) là sự kết hợp giữa Green và Blue, nên set kênh Red =0
                    Cyan.SetPixel(x,y, Color.FromArgb(0, G, B));

                    //Magenta (tím) là sự kết hợp giữa Red và Blue, nên set kênh Green =0
                    Magenta.SetPixel(x, y, Color.FromArgb(R, 0, B));

                    //Yellow(vàng) là sự kết hợp giữa Red và Green, nên set kênh Blue = 0
                    Yellow.SetPixel(x, y, Color.FromArgb(R, G, 0));

                    //Black (Đen) là MIN (R,G,B)
                    byte K = Math.Min(R, Math.Min(G, B));
                    Black.SetPixel(x, y, Color.FromArgb(K, K, K));

                }
            // add các hình tương ứng các kênh màu C-M-Y-K vào list
            // do List là kiểu dữ liệu mảng không cần biết trước kích thước nên ta có thể add các element của List 
            // mà không sợ bị tràn kích thước (ưu điểm của C# so với C++)
            CMYK.Add(Cyan);
            CMYK.Add(Magenta);
            CMYK.Add(Yellow);
            CMYK.Add(Black);

            //Hàm trả về 1 list 4 ảnh bitmap tương ứng 4 kênh màu C-M-Y-K
            return CMYK;


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
    }
}
