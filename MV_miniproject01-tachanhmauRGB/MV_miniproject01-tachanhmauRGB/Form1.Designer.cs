namespace MV_miniproject01_tachanhmauRGB
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.picBox_Hinhgoc = new System.Windows.Forms.PictureBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.picBox_RED = new System.Windows.Forms.PictureBox();
            this.label3 = new System.Windows.Forms.Label();
            this.picBox_GREEN = new System.Windows.Forms.PictureBox();
            this.label4 = new System.Windows.Forms.Label();
            this.picBox_BLUE = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_Hinhgoc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_RED)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_GREEN)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_BLUE)).BeginInit();
            this.SuspendLayout();
            // 
            // picBox_Hinhgoc
            // 
            this.picBox_Hinhgoc.Location = new System.Drawing.Point(142, 7);
            this.picBox_Hinhgoc.Name = "picBox_Hinhgoc";
            this.picBox_Hinhgoc.Size = new System.Drawing.Size(512, 512);
            this.picBox_Hinhgoc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBox_Hinhgoc.TabIndex = 0;
            this.picBox_Hinhgoc.TabStop = false;
            this.picBox_Hinhgoc.Click += new System.EventHandler(this.pictureBox1_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F);
            this.label1.Location = new System.Drawing.Point(1, 237);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(123, 20);
            this.label1.TabIndex = 1;
            this.label1.Text = "Hình GRB gốc:";
            this.label1.Click += new System.EventHandler(this.label1_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F);
            this.label2.Location = new System.Drawing.Point(660, 237);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(130, 20);
            this.label2.TabIndex = 3;
            this.label2.Text = "Kênh màu RED:";
            this.label2.Click += new System.EventHandler(this.label2_Click);
            // 
            // picBox_RED
            // 
            this.picBox_RED.Location = new System.Drawing.Point(812, 7);
            this.picBox_RED.Name = "picBox_RED";
            this.picBox_RED.Size = new System.Drawing.Size(512, 512);
            this.picBox_RED.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBox_RED.TabIndex = 2;
            this.picBox_RED.TabStop = false;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F);
            this.label3.Location = new System.Drawing.Point(1, 775);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(129, 17);
            this.label3.TabIndex = 5;
            this.label3.Text = "Kênh màu GREEN:";
            // 
            // picBox_GREEN
            // 
            this.picBox_GREEN.Location = new System.Drawing.Point(142, 531);
            this.picBox_GREEN.Name = "picBox_GREEN";
            this.picBox_GREEN.Size = new System.Drawing.Size(512, 512);
            this.picBox_GREEN.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBox_GREEN.TabIndex = 4;
            this.picBox_GREEN.TabStop = false;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.label4.Location = new System.Drawing.Point(668, 775);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(122, 18);
            this.label4.TabIndex = 7;
            this.label4.Text = "Kênh màu BLUE:";
            // 
            // picBox_BLUE
            // 
            this.picBox_BLUE.Location = new System.Drawing.Point(812, 525);
            this.picBox_BLUE.Name = "picBox_BLUE";
            this.picBox_BLUE.Size = new System.Drawing.Size(512, 512);
            this.picBox_BLUE.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBox_BLUE.TabIndex = 6;
            this.picBox_BLUE.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1336, 1055);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.picBox_BLUE);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.picBox_GREEN);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.picBox_RED);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.picBox_Hinhgoc);
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Tách ảnh màu RGB cho từng kênh màu";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            ((System.ComponentModel.ISupportInitialize)(this.picBox_Hinhgoc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_RED)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_GREEN)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBox_BLUE)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox picBox_Hinhgoc;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.PictureBox picBox_RED;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.PictureBox picBox_GREEN;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.PictureBox picBox_BLUE;
    }
}

