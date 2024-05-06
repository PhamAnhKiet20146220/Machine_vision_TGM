namespace Grayscale_Lightness
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
            this.picBoxHinhGoc = new System.Windows.Forms.PictureBox();
            this.lblHinhGoc = new System.Windows.Forms.Label();
            this.picBoxHinhxamLightness = new System.Windows.Forms.PictureBox();
            this.lblHinhXamLightness = new System.Windows.Forms.Label();
            this.picBoxHinhXamAverage = new System.Windows.Forms.PictureBox();
            this.lblHinhXamAverage = new System.Windows.Forms.Label();
            this.picBoxHinhXamLuminance = new System.Windows.Forms.PictureBox();
            this.lblHinhXamLuminance = new System.Windows.Forms.Label();
            this.picBoxHinhNhiPhan = new System.Windows.Forms.PictureBox();
            this.lblHinhNhiPhan = new System.Windows.Forms.Label();
            this.hScrollBarNhiPhan = new System.Windows.Forms.HScrollBar();
            this.lblthreshold = new System.Windows.Forms.Label();
            this.lblNguong = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhxamLightness)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhXamAverage)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhXamLuminance)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhNhiPhan)).BeginInit();
            this.SuspendLayout();
            // 
            // picBoxHinhGoc
            // 
            this.picBoxHinhGoc.Location = new System.Drawing.Point(27, 32);
            this.picBoxHinhGoc.Name = "picBoxHinhGoc";
            this.picBoxHinhGoc.Size = new System.Drawing.Size(500, 381);
            this.picBoxHinhGoc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhGoc.TabIndex = 0;
            this.picBoxHinhGoc.TabStop = false;
            // 
            // lblHinhGoc
            // 
            this.lblHinhGoc.AutoSize = true;
            this.lblHinhGoc.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhGoc.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhGoc.Location = new System.Drawing.Point(22, 3);
            this.lblHinhGoc.Name = "lblHinhGoc";
            this.lblHinhGoc.Size = new System.Drawing.Size(104, 26);
            this.lblHinhGoc.TabIndex = 1;
            this.lblHinhGoc.Text = "Hình gốc:";
            this.lblHinhGoc.Click += new System.EventHandler(this.lblHinhGoc_Click);
            // 
            // picBoxHinhxamLightness
            // 
            this.picBoxHinhxamLightness.Location = new System.Drawing.Point(561, 32);
            this.picBoxHinhxamLightness.Name = "picBoxHinhxamLightness";
            this.picBoxHinhxamLightness.Size = new System.Drawing.Size(500, 381);
            this.picBoxHinhxamLightness.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhxamLightness.TabIndex = 2;
            this.picBoxHinhxamLightness.TabStop = false;
            // 
            // lblHinhXamLightness
            // 
            this.lblHinhXamLightness.AutoSize = true;
            this.lblHinhXamLightness.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhXamLightness.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhXamLightness.Location = new System.Drawing.Point(556, 3);
            this.lblHinhXamLightness.Name = "lblHinhXamLightness";
            this.lblHinhXamLightness.Size = new System.Drawing.Size(258, 26);
            this.lblHinhXamLightness.TabIndex = 3;
            this.lblHinhXamLightness.Text = "Hình mức xám Lightness:";
            this.lblHinhXamLightness.Click += new System.EventHandler(this.label1_Click);
            // 
            // picBoxHinhXamAverage
            // 
            this.picBoxHinhXamAverage.Location = new System.Drawing.Point(27, 446);
            this.picBoxHinhXamAverage.Name = "picBoxHinhXamAverage";
            this.picBoxHinhXamAverage.Size = new System.Drawing.Size(500, 381);
            this.picBoxHinhXamAverage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhXamAverage.TabIndex = 4;
            this.picBoxHinhXamAverage.TabStop = false;
            // 
            // lblHinhXamAverage
            // 
            this.lblHinhXamAverage.AutoSize = true;
            this.lblHinhXamAverage.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhXamAverage.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhXamAverage.Location = new System.Drawing.Point(22, 417);
            this.lblHinhXamAverage.Name = "lblHinhXamAverage";
            this.lblHinhXamAverage.Size = new System.Drawing.Size(246, 26);
            this.lblHinhXamAverage.TabIndex = 5;
            this.lblHinhXamAverage.Text = "Hình mức xám Average:";
            this.lblHinhXamAverage.Click += new System.EventHandler(this.lblHinhXamAvergare_Click);
            // 
            // picBoxHinhXamLuminance
            // 
            this.picBoxHinhXamLuminance.Location = new System.Drawing.Point(561, 446);
            this.picBoxHinhXamLuminance.Name = "picBoxHinhXamLuminance";
            this.picBoxHinhXamLuminance.Size = new System.Drawing.Size(500, 381);
            this.picBoxHinhXamLuminance.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhXamLuminance.TabIndex = 6;
            this.picBoxHinhXamLuminance.TabStop = false;
            // 
            // lblHinhXamLuminance
            // 
            this.lblHinhXamLuminance.AutoSize = true;
            this.lblHinhXamLuminance.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhXamLuminance.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhXamLuminance.Location = new System.Drawing.Point(556, 417);
            this.lblHinhXamLuminance.Name = "lblHinhXamLuminance";
            this.lblHinhXamLuminance.Size = new System.Drawing.Size(272, 26);
            this.lblHinhXamLuminance.TabIndex = 7;
            this.lblHinhXamLuminance.Text = "Hình mức xám Luminance:";
            // 
            // picBoxHinhNhiPhan
            // 
            this.picBoxHinhNhiPhan.Location = new System.Drawing.Point(1089, 32);
            this.picBoxHinhNhiPhan.Name = "picBoxHinhNhiPhan";
            this.picBoxHinhNhiPhan.Size = new System.Drawing.Size(500, 381);
            this.picBoxHinhNhiPhan.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhNhiPhan.TabIndex = 8;
            this.picBoxHinhNhiPhan.TabStop = false;
            this.picBoxHinhNhiPhan.Click += new System.EventHandler(this.pictureBox1_Click);
            // 
            // lblHinhNhiPhan
            // 
            this.lblHinhNhiPhan.AutoSize = true;
            this.lblHinhNhiPhan.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhNhiPhan.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhNhiPhan.Location = new System.Drawing.Point(1084, 3);
            this.lblHinhNhiPhan.Name = "lblHinhNhiPhan";
            this.lblHinhNhiPhan.Size = new System.Drawing.Size(152, 26);
            this.lblHinhNhiPhan.TabIndex = 9;
            this.lblHinhNhiPhan.Text = "Hình nhị phân:";
            // 
            // hScrollBarNhiPhan
            // 
            this.hScrollBarNhiPhan.Location = new System.Drawing.Point(1181, 422);
            this.hScrollBarNhiPhan.Maximum = 255;
            this.hScrollBarNhiPhan.Name = "hScrollBarNhiPhan";
            this.hScrollBarNhiPhan.Size = new System.Drawing.Size(408, 34);
            this.hScrollBarNhiPhan.TabIndex = 11;
            this.hScrollBarNhiPhan.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hScrollBarNhiPhan_Scroll);
            // 
            // lblthreshold
            // 
            this.lblthreshold.AutoSize = true;
            this.lblthreshold.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblthreshold.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblthreshold.Location = new System.Drawing.Point(1084, 422);
            this.lblthreshold.Name = "lblthreshold";
            this.lblthreshold.Size = new System.Drawing.Size(94, 26);
            this.lblthreshold.TabIndex = 12;
            this.lblthreshold.Text = "Ngưỡng:";
            this.lblthreshold.Click += new System.EventHandler(this.label1_Click_1);
            // 
            // lblNguong
            // 
            this.lblNguong.AutoSize = true;
            this.lblNguong.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblNguong.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblNguong.Location = new System.Drawing.Point(1313, 456);
            this.lblNguong.Name = "lblNguong";
            this.lblNguong.Size = new System.Drawing.Size(0, 26);
            this.lblNguong.TabIndex = 13;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1682, 839);
            this.Controls.Add(this.lblNguong);
            this.Controls.Add(this.lblthreshold);
            this.Controls.Add(this.hScrollBarNhiPhan);
            this.Controls.Add(this.lblHinhNhiPhan);
            this.Controls.Add(this.picBoxHinhNhiPhan);
            this.Controls.Add(this.lblHinhXamLuminance);
            this.Controls.Add(this.picBoxHinhXamLuminance);
            this.Controls.Add(this.lblHinhXamAverage);
            this.Controls.Add(this.picBoxHinhXamAverage);
            this.Controls.Add(this.lblHinhXamLightness);
            this.Controls.Add(this.picBoxHinhxamLightness);
            this.Controls.Add(this.lblHinhGoc);
            this.Controls.Add(this.picBoxHinhGoc);
            this.Name = "Form1";
            this.Text = " Chuyển ảnh màu RGB sang ảnh mức xám Grayscale";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhxamLightness)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhXamAverage)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhXamLuminance)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhNhiPhan)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox picBoxHinhGoc;
        private System.Windows.Forms.Label lblHinhGoc;
        private System.Windows.Forms.PictureBox picBoxHinhxamLightness;
        private System.Windows.Forms.Label lblHinhXamLightness;
        private System.Windows.Forms.PictureBox picBoxHinhXamAverage;
        private System.Windows.Forms.Label lblHinhXamAverage;
        private System.Windows.Forms.PictureBox picBoxHinhXamLuminance;
        private System.Windows.Forms.Label lblHinhXamLuminance;
        private System.Windows.Forms.PictureBox picBoxHinhNhiPhan;
        private System.Windows.Forms.Label lblHinhNhiPhan;
        private System.Windows.Forms.HScrollBar hScrollBarNhiPhan;
        private System.Windows.Forms.Label lblthreshold;
        private System.Windows.Forms.Label lblNguong;
    }
}

