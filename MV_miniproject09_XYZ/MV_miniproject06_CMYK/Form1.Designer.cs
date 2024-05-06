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
            this.pictBoxH = new System.Windows.Forms.PictureBox();
            this.lblHue = new System.Windows.Forms.Label();
            this.picBoxS = new System.Windows.Forms.PictureBox();
            this.lblSaturation = new System.Windows.Forms.Label();
            this.picBoxI = new System.Windows.Forms.PictureBox();
            this.lblIntensity = new System.Windows.Forms.Label();
            this.picBoxHSI = new System.Windows.Forms.PictureBox();
            this.lblHSI = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxH)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxS)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxI)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHSI)).BeginInit();
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
            this.picBoxHinhGoc.Size = new System.Drawing.Size(256, 256);
            this.picBoxHinhGoc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHinhGoc.TabIndex = 1;
            this.picBoxHinhGoc.TabStop = false;
            // 
            // pictBoxH
            // 
            this.pictBoxH.Location = new System.Drawing.Point(14, 335);
            this.pictBoxH.Name = "pictBoxH";
            this.pictBoxH.Size = new System.Drawing.Size(256, 256);
            this.pictBoxH.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictBoxH.TabIndex = 3;
            this.pictBoxH.TabStop = false;
            this.pictBoxH.Click += new System.EventHandler(this.pictBoxC_Click);
            // 
            // lblHue
            // 
            this.lblHue.AutoSize = true;
            this.lblHue.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHue.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHue.Location = new System.Drawing.Point(9, 307);
            this.lblHue.Name = "lblHue";
            this.lblHue.Size = new System.Drawing.Size(84, 25);
            this.lblHue.TabIndex = 2;
            this.lblHue.Text = "Kênh X:";
            this.lblHue.Click += new System.EventHandler(this.lblHue_Click);
            // 
            // picBoxS
            // 
            this.picBoxS.Location = new System.Drawing.Point(307, 335);
            this.picBoxS.Name = "picBoxS";
            this.picBoxS.Size = new System.Drawing.Size(256, 256);
            this.picBoxS.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxS.TabIndex = 5;
            this.picBoxS.TabStop = false;
            // 
            // lblSaturation
            // 
            this.lblSaturation.AutoSize = true;
            this.lblSaturation.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSaturation.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblSaturation.Location = new System.Drawing.Point(302, 307);
            this.lblSaturation.Name = "lblSaturation";
            this.lblSaturation.Size = new System.Drawing.Size(83, 25);
            this.lblSaturation.TabIndex = 4;
            this.lblSaturation.Text = "Kênh Y:";
            // 
            // picBoxI
            // 
            this.picBoxI.Location = new System.Drawing.Point(599, 335);
            this.picBoxI.Name = "picBoxI";
            this.picBoxI.Size = new System.Drawing.Size(256, 256);
            this.picBoxI.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxI.TabIndex = 7;
            this.picBoxI.TabStop = false;
            this.picBoxI.Click += new System.EventHandler(this.picBoxY_Click);
            // 
            // lblIntensity
            // 
            this.lblIntensity.AutoSize = true;
            this.lblIntensity.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblIntensity.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblIntensity.Location = new System.Drawing.Point(594, 307);
            this.lblIntensity.Name = "lblIntensity";
            this.lblIntensity.Size = new System.Drawing.Size(82, 25);
            this.lblIntensity.TabIndex = 6;
            this.lblIntensity.Text = "Kênh Z:";
            this.lblIntensity.Click += new System.EventHandler(this.lblIntensity_Click);
            // 
            // picBoxHSI
            // 
            this.picBoxHSI.Location = new System.Drawing.Point(892, 335);
            this.picBoxHSI.Name = "picBoxHSI";
            this.picBoxHSI.Size = new System.Drawing.Size(256, 256);
            this.picBoxHSI.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxHSI.TabIndex = 9;
            this.picBoxHSI.TabStop = false;
            // 
            // lblHSI
            // 
            this.lblHSI.AutoSize = true;
            this.lblHSI.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHSI.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblHSI.Location = new System.Drawing.Point(887, 307);
            this.lblHSI.Name = "lblHSI";
            this.lblHSI.Size = new System.Drawing.Size(102, 25);
            this.lblHSI.TabIndex = 8;
            this.lblHSI.Text = "Hình XYZ:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1176, 606);
            this.Controls.Add(this.picBoxHSI);
            this.Controls.Add(this.lblHSI);
            this.Controls.Add(this.picBoxI);
            this.Controls.Add(this.lblIntensity);
            this.Controls.Add(this.picBoxS);
            this.Controls.Add(this.lblSaturation);
            this.Controls.Add(this.pictBoxH);
            this.Controls.Add(this.lblHue);
            this.Controls.Add(this.picBoxHinhGoc);
            this.Controls.Add(this.lblHinhGoc);
            this.Name = "Form1";
            this.Text = "Chuyển đổi không gian màu RGB sang HSI";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxH)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxS)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxI)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHSI)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblHinhGoc;
        private System.Windows.Forms.PictureBox picBoxHinhGoc;
        private System.Windows.Forms.PictureBox pictBoxH;
        private System.Windows.Forms.Label lblHue;
        private System.Windows.Forms.PictureBox picBoxS;
        private System.Windows.Forms.Label lblSaturation;
        private System.Windows.Forms.PictureBox picBoxI;
        private System.Windows.Forms.Label lblIntensity;
        private System.Windows.Forms.PictureBox picBoxHSI;
        private System.Windows.Forms.Label lblHSI;
    }
}

