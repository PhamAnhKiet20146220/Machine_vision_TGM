namespace MV_miniproject06_CMYK
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
            this.lblHinhGoc = new System.Windows.Forms.Label();
            this.picBoxHinhGoc = new System.Windows.Forms.PictureBox();
            this.pictBoxC = new System.Windows.Forms.PictureBox();
            this.lblCyan = new System.Windows.Forms.Label();
            this.lblthreshold = new System.Windows.Forms.Label();
            this.lblNguong = new System.Windows.Forms.Label();
            this.hScrollBarNhiPhan = new System.Windows.Forms.HScrollBar();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxC)).BeginInit();
            this.SuspendLayout();
            // 
            // lblHinhGoc
            // 
            this.lblHinhGoc.AutoSize = true;
            this.lblHinhGoc.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHinhGoc.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHinhGoc.Location = new System.Drawing.Point(12, 9);
            this.lblHinhGoc.Name = "lblHinhGoc";
            this.lblHinhGoc.Size = new System.Drawing.Size(141, 25);
            this.lblHinhGoc.TabIndex = 0;
            this.lblHinhGoc.Text = "Hình RGB gốc:";
            this.lblHinhGoc.Click += new System.EventHandler(this.label1_Click);
            // 
            // picBoxHinhGoc
            // 
            this.picBoxHinhGoc.Location = new System.Drawing.Point(17, 37);
            this.picBoxHinhGoc.Name = "picBoxHinhGoc";
            this.picBoxHinhGoc.Size = new System.Drawing.Size(512, 512);
            this.picBoxHinhGoc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhGoc.TabIndex = 1;
            this.picBoxHinhGoc.TabStop = false;
            this.picBoxHinhGoc.Click += new System.EventHandler(this.picBoxHinhGoc_Click);
            // 
            // pictBoxC
            // 
            this.pictBoxC.Location = new System.Drawing.Point(626, 37);
            this.pictBoxC.Name = "pictBoxC";
            this.pictBoxC.Size = new System.Drawing.Size(512, 512);
            this.pictBoxC.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictBoxC.TabIndex = 3;
            this.pictBoxC.TabStop = false;
            // 
            // lblCyan
            // 
            this.lblCyan.AutoSize = true;
            this.lblCyan.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCyan.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblCyan.Location = new System.Drawing.Point(621, 9);
            this.lblCyan.Name = "lblCyan";
            this.lblCyan.Size = new System.Drawing.Size(325, 25);
            this.lblCyan.TabIndex = 2;
            this.lblCyan.Text = "Hình sau khi nhận dạng đường biên:";
            this.lblCyan.Click += new System.EventHandler(this.lblCyan_Click);
            // 
            // lblthreshold
            // 
            this.lblthreshold.AutoSize = true;
            this.lblthreshold.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblthreshold.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblthreshold.Location = new System.Drawing.Point(12, 571);
            this.lblthreshold.Name = "lblthreshold";
            this.lblthreshold.Size = new System.Drawing.Size(94, 26);
            this.lblthreshold.TabIndex = 13;
            this.lblthreshold.Text = "Ngưỡng:";
            // 
            // lblNguong
            // 
            this.lblNguong.AutoSize = true;
            this.lblNguong.Font = new System.Drawing.Font("Microsoft Sans Serif", 13F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblNguong.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblNguong.Location = new System.Drawing.Point(112, 571);
            this.lblNguong.Name = "lblNguong";
            this.lblNguong.Size = new System.Drawing.Size(0, 26);
            this.lblNguong.TabIndex = 14;
            // 
            // hScrollBarNhiPhan
            // 
            this.hScrollBarNhiPhan.Location = new System.Drawing.Point(17, 607);
            this.hScrollBarNhiPhan.Maximum = 255;
            this.hScrollBarNhiPhan.Name = "hScrollBarNhiPhan";
            this.hScrollBarNhiPhan.Size = new System.Drawing.Size(1121, 34);
            this.hScrollBarNhiPhan.TabIndex = 15;
            this.hScrollBarNhiPhan.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hScrollBarNhiPhan_Scroll);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1176, 683);
            this.Controls.Add(this.hScrollBarNhiPhan);
            this.Controls.Add(this.lblNguong);
            this.Controls.Add(this.lblthreshold);
            this.Controls.Add(this.pictBoxC);
            this.Controls.Add(this.lblCyan);
            this.Controls.Add(this.picBoxHinhGoc);
            this.Controls.Add(this.lblHinhGoc);
            this.Name = "Form1";
            this.Text = "Chuyển đổi không gian màu RGB sang CMYK";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxC)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblHinhGoc;
        private System.Windows.Forms.PictureBox picBoxHinhGoc;
        private System.Windows.Forms.PictureBox pictBoxC;
        private System.Windows.Forms.Label lblCyan;
        private System.Windows.Forms.Label lblthreshold;
        private System.Windows.Forms.Label lblNguong;
        private System.Windows.Forms.HScrollBar hScrollBarNhiPhan;
    }
}

