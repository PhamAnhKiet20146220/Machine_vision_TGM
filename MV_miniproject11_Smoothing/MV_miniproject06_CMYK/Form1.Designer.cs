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
            this.picBoxM = new System.Windows.Forms.PictureBox();
            this.lblMagenta = new System.Windows.Forms.Label();
            this.picBoxY = new System.Windows.Forms.PictureBox();
            this.lblYellow = new System.Windows.Forms.Label();
            this.picBoxB = new System.Windows.Forms.PictureBox();
            this.lblBlack = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxC)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxM)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxY)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxB)).BeginInit();
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
            // pictBoxC
            // 
            this.pictBoxC.Location = new System.Drawing.Point(14, 335);
            this.pictBoxC.Name = "pictBoxC";
            this.pictBoxC.Size = new System.Drawing.Size(256, 256);
            this.pictBoxC.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictBoxC.TabIndex = 3;
            this.pictBoxC.TabStop = false;
            // 
            // lblCyan
            // 
            this.lblCyan.AutoSize = true;
            this.lblCyan.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCyan.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblCyan.Location = new System.Drawing.Point(9, 307);
            this.lblCyan.Name = "lblCyan";
            this.lblCyan.Size = new System.Drawing.Size(239, 25);
            this.lblCyan.TabIndex = 2;
            this.lblCyan.Text = "Hình làm mượt (mask 3x3)";
            this.lblCyan.Click += new System.EventHandler(this.lblCyan_Click);
            // 
            // picBoxM
            // 
            this.picBoxM.Location = new System.Drawing.Point(307, 335);
            this.picBoxM.Name = "picBoxM";
            this.picBoxM.Size = new System.Drawing.Size(256, 256);
            this.picBoxM.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxM.TabIndex = 5;
            this.picBoxM.TabStop = false;
            // 
            // lblMagenta
            // 
            this.lblMagenta.AutoSize = true;
            this.lblMagenta.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblMagenta.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblMagenta.Location = new System.Drawing.Point(302, 307);
            this.lblMagenta.Name = "lblMagenta";
            this.lblMagenta.Size = new System.Drawing.Size(239, 25);
            this.lblMagenta.TabIndex = 4;
            this.lblMagenta.Text = "Hình làm mượt (mask 5x5)";
            // 
            // picBoxY
            // 
            this.picBoxY.Location = new System.Drawing.Point(599, 335);
            this.picBoxY.Name = "picBoxY";
            this.picBoxY.Size = new System.Drawing.Size(256, 256);
            this.picBoxY.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxY.TabIndex = 7;
            this.picBoxY.TabStop = false;
            this.picBoxY.Click += new System.EventHandler(this.picBoxY_Click);
            // 
            // lblYellow
            // 
            this.lblYellow.AutoSize = true;
            this.lblYellow.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblYellow.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblYellow.Location = new System.Drawing.Point(594, 307);
            this.lblYellow.Name = "lblYellow";
            this.lblYellow.Size = new System.Drawing.Size(239, 25);
            this.lblYellow.TabIndex = 6;
            this.lblYellow.Text = "Hình làm mượt (mask 7x7)";
            // 
            // picBoxB
            // 
            this.picBoxB.Location = new System.Drawing.Point(892, 335);
            this.picBoxB.Name = "picBoxB";
            this.picBoxB.Size = new System.Drawing.Size(256, 256);
            this.picBoxB.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.picBoxB.TabIndex = 9;
            this.picBoxB.TabStop = false;
            // 
            // lblBlack
            // 
            this.lblBlack.AutoSize = true;
            this.lblBlack.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblBlack.ForeColor = System.Drawing.Color.OrangeRed;
            this.lblBlack.Location = new System.Drawing.Point(887, 307);
            this.lblBlack.Name = "lblBlack";
            this.lblBlack.Size = new System.Drawing.Size(239, 25);
            this.lblBlack.TabIndex = 8;
            this.lblBlack.Text = "Hình làm mượt (mask 9x9)";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1176, 606);
            this.Controls.Add(this.picBoxB);
            this.Controls.Add(this.lblBlack);
            this.Controls.Add(this.picBoxY);
            this.Controls.Add(this.lblYellow);
            this.Controls.Add(this.picBoxM);
            this.Controls.Add(this.lblMagenta);
            this.Controls.Add(this.pictBoxC);
            this.Controls.Add(this.lblCyan);
            this.Controls.Add(this.picBoxHinhGoc);
            this.Controls.Add(this.lblHinhGoc);
            this.Name = "Form1";
            this.Text = "Chuyển đổi không gian màu RGB sang CMYK";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picBoxHinhGoc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictBoxC)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxM)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxY)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picBoxB)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblHinhGoc;
        private System.Windows.Forms.PictureBox picBoxHinhGoc;
        private System.Windows.Forms.PictureBox pictBoxC;
        private System.Windows.Forms.Label lblCyan;
        private System.Windows.Forms.PictureBox picBoxM;
        private System.Windows.Forms.Label lblMagenta;
        private System.Windows.Forms.PictureBox picBoxY;
        private System.Windows.Forms.Label lblYellow;
        private System.Windows.Forms.PictureBox picBoxB;
        private System.Windows.Forms.Label lblBlack;
    }
}

