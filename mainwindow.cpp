#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QtMath>
#include <QDateTime>
#include <qrgb.h>
#include <QTimer>
#include <QSpacerItem>
#include <thread>
#include <QDebug>
#include <QElapsedTimer>
#define CONCAT_(x,y) x##y
#define CONCAT(x,y) CONCAT_(x,y)

#define CHECKTIME(x)  \
QElapsedTimer CONCAT(sb_, __LINE__); \
    CONCAT(sb_, __LINE__).start(); \
    x \
    qDebug() << __FUNCTION__ << ":" << __LINE__ << " Elapsed time: " <<  CONCAT(sb_, __LINE__).elapsed() << " ms.";


#define TROW 3
#define TCOL 3
#define TROW2 5
#define TCOL2 5

#define LSBYTE 0x000000ff

//dummy data to test QImg RGB ordering
alignas(alignof(unsigned int)) static uchar dbg[] = {
    0xde,0xca,0xfb,0xad,
    0xff,0x00,0x00,0x00,
    0xff,0x00,0x00,0x00,
    0xff,0x00,0x00,0x00,
    0xff,0x00,0x00,0x00,
    0xff,0x00,0x00,0x00
};

//    QImage t {dbg, 2,3, QImage::Format::Format_RGB32};
//    Bits<unsigned int> tdbg = {.val = t.pixel(0,0)}; //   to_gray(ref, data)



inline int get_at(int x, int y, int rows, const int err = 256 * 256)
{
    int result = y * rows + x;
    if (result >= err)
        return x;
    return result;
}

unsigned npow2(unsigned x)
{
    x--;
    x |= (x>>1);
    x |= (x>>2);
    x |= (x>>4);
    x |= (x>>8);
    x |= (x>>16);
    return ++x;
}

template <typename T>
T clampnf(T val, const T near, const T far) {
    if (val < near) return near;
    else if (val > far) return far;
    else return val;
}

template <typename T>
union Bits
{
    T val;
    char data[sizeof(T)];
};

//1,2,3,4,5
//1,2,3,4,5
//1,2,3,4,5
//1,2,3,4,5
//1,2,3,4,5

struct kernel_t
{
    float val3x3[9];
    float val5x5[25];

    kernel_t()
    {
        memset(&val3x3, 0, sizeof(val3x3));
        memset(&val5x5, 0, sizeof(val5x5));
    }

    kernel_t(float (&data)[9]) {
        memcpy(val3x3, data, sizeof(data));
    }


    FRGB operator * (FRGB (&data)[25])
    {
        FRGB out;
        memset(&out , 0, sizeof(out));
        float sum =0.0f;
        int res = 0;

        for(int i=0; i < 25; i++)
            sum += val5x5[i];
        for(int i=0; i < 3; i++) {
            for(int ii=0; ii < 25; ii++) {
                data[ii].rgb[i] *= val5x5[ii];
                res += data[ii].rgb[i];
            }
            out.rgb[i] = res;
            res = 0;

        }
        return out;
    }

    FRGB operator * (FRGB (&data)[9])
    {
        FRGB out;
        memset(&out , 0, sizeof(out));
        float sum =0.0f;
        int res = 0;

        for(int i=0; i < 9; i++)
            sum += val3x3[i];
        for(int i=0; i < 3; i++) {
            for(int ii=0; ii < 9; ii++) {             // 1, 2, 3,       1, 1,1,
                data[ii].rgb[i] *= val3x3[ii];          // 1, 2, 3,   x   1, 1,1,
                res += data[ii].rgb[i];
            }
            out.rgb[i] = res;
            res = 0;

        }
        return out;
    }
};

static float _sharp[] = {        0.0f,-1.0f,0.0f,
                                -1.0f,5.0f,-1.0f,
                                 0.0f,-1.0f,0.0f};

static float _topsobel[] = {1.0f, 2.0f,1.0f,
                            0.0f,0.0f,0.0f,
                           -1.0f,-2.0f,-1.0f};

static float _blur[] = {    0.111f, 0.111f, 0.111f,
                            0.111f, 0.111f, 0.111f,
                            0.111f, 0.111f, 0.111f};

static float _ident[] = {0.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f };

static  kernel_t gaussianConv{_blur};

static kernel_t sharpConv{_sharp};

static kernel_t iSharpConv {_topsobel};

static kernel_t identConv {_ident};



void MainWindow::worker_helper::doWork()
{
    m_worker = std::thread  {[this]()
    {
        int total_len = p_Parent->m_rgbdata.size();

    if (m_type != eConvType::Custom5x5) {
        while (m_count-- >= 0)
            for(int y = m_y ;y < m_h  ; y++ ) {
                for(int x = m_x; x < m_w ; x++)
                {
                    FRGB newpix;
                    for(int i=0; i < TROW; i++) {
                        for(int j=0; j < TCOL; j++) {
                            m_localCtx.rgbchans[i*TROW+j] = p_Parent->m_rgbdata.at(get_at(x+j,y+i, p_Parent->m_currImg.width(), total_len));
                            m_localCtx.fRGB[i*TROW +j].rgb[0] = ((m_localCtx.rgbchans[i*1 +j] >> 16) & LSBYTE);// / 255.0f;
                            m_localCtx.fRGB[i*TROW +j].rgb[1] = ((m_localCtx.rgbchans[i*1 +j] >> 8) & LSBYTE);// / 255.0f;
                            m_localCtx.fRGB[i*TROW +j].rgb[2] = (m_localCtx.rgbchans[i*1 +j] & LSBYTE);// / 255.0f;
                        }
                    }
                    switch (m_type) {
                    case eConvType::GaussianBlur:
                        newpix= gaussianConv * m_localCtx.fRGB;
                        break;
                    case eConvType::Sharper:
                        newpix = sharpConv * m_localCtx.fRGB;
                        break;
                    case eConvType::IntensivSharper:
                        newpix = iSharpConv * m_localCtx.fRGB;
                        break;
                    case eConvType::Identity:
                        newpix = identConv * m_localCtx.fRGB;
                        break;
                    case eConvType::Custom3x3:
                        newpix = *(p_Parent->pCustKernel) * m_localCtx.fRGB;
                        break;
                    default:
                        break;
                    }
                    QRgb t = qRgba(newpix.rgb[0] , newpix.rgb[1], newpix.rgb[2] ,
                                   qAlpha(m_localCtx.rgbchans[4]));
                    p_Parent->m_rgbdata[get_at(x,y, p_Parent->m_currImg.width(), total_len)] = t;

                }
            }
    } else {
        while (m_count-- >= 0)
            for(int y = m_y; y < m_h ; y++ ) {
                for(int x = m_x; x < m_w; x++)
                {
                    FRGB newpix;

                    for(int i=0; i < TROW2; i++) {
                        for(int j=0; j < TCOL2; j++) {
                            m_localCtx.rgbchans2[i*TROW2+j] = p_Parent->m_rgbdata.at(get_at(x+j,y+i, p_Parent->m_currImg.width(),total_len));
                            m_localCtx.fRGB2[i*TROW2 +j].rgb[0] = ((m_localCtx.rgbchans2[i*1 +j] >> 16) & LSBYTE);// / 255.0f;
                            m_localCtx.fRGB2[i*TROW2 +j].rgb[1] = ((m_localCtx.rgbchans2[i*1 +j] >> 8) & LSBYTE);// / 255.0f;
                            m_localCtx.fRGB2[i*TROW2 +j].rgb[2] = (m_localCtx.rgbchans2[i*1 +j] & LSBYTE);// / 255.0f;
                        }
                    }
                    newpix = *(p_Parent->pCustKernel) * m_localCtx.fRGB2;

                    QRgb t = qRgba(newpix.rgb[0] , newpix.rgb[1], newpix.rgb[2] ,
                                   qAlpha(m_localCtx.rgbchans2[4]));

                    p_Parent->m_rgbdata[get_at(x,y, p_Parent->m_currImg.width(), total_len)] = t;


                }
            }
        }
        }//thread
    };

}

void MainWindow::worker_helper::doWorkAccel()
{

}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow),
    m_val{1},
    m_mtenabled{false}, m_gpuAccel{false},
    pCustKernel{nullptr},
    m_imgScreen{parent},
    m_warmer{&m_gpukern}
 {
    ui->setupUi(this);
    for(int i=0; i < 3; i++) m_3x3layout[i] = new QVBoxLayout{parent};
    for(int i=0; i < 5; i++) m_5x5layout[i] = new QVBoxLayout{parent};

    for(int i=0; i < 9; i++) m_spin3x3[i] = new QDoubleSpinBox{parent};
    for(int i=0; i < 25; i++) m_spin5x5[i] = new QDoubleSpinBox{parent};

    memset(&m_rgbctx, 0 , sizeof(m_rgbctx));
    pCustKernel = new kernel_t;
//    ui->grayscale->setDisabled(true);
//    ui->grayscale->setDown(true);

//    ui->pushButton->setEnabled(false);
    ui->gaussianBlur->setEnabled(false);
    ui->intensiveSharper->setEnabled(false);
    ui->sharper->setEnabled(false);
    ui->intensiveSharper->setEnabled(false);
    ui->grayscale->setEnabled(false);
    ui->custKern->setEnabled(false);
    ui->identityBtn->setEnabled(false);
    ui->original->setEnabled(false);


    m_imgScreen.setBackgroundRole(QPalette::Base);
//    m_imgScreen.setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//    m_imgScreen.setScaledContents(true);


    ui->scrollArea->setWidget(&m_imgScreen);
    ui->scrollArea->setWidgetResizable(true);
    ui->scrollArea->ensureWidgetVisible(&m_imgScreen);
    ui->scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAsNeeded);
    ui->scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAsNeeded);

    //kern 3x3 ui
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            m_spin3x3[get_at(j, i, 3, 9)]->setDecimals(6);
            m_spin3x3[get_at(j, i, 3, 9)]->setMaximum(99.999999);
            m_spin3x3[get_at(j, i, 3, 9)]->setMinimum(-99.999999);
            m_spin3x3[get_at(j, i, 3, 9)]->setMinimumSize(QSize{90,24});
        }
    }

    for(int y=0; y < 3; y++) {

        for(int x = 0 ; x < 3; x++) {
            m_3x3layout[y]->addWidget(m_spin3x3[get_at(x, y, 3, 9)]);
            m_3x3layout[y]->setSizeConstraint(QLayout::SetDefaultConstraint);
        }
        ui->horizontalLayout_3x3->addLayout(m_3x3layout[y]);
//        ui->horizontalLayout_3x3->addSpacerItem(&s);
    }

    //kern 5x5 ui
    for(int i=0; i < 5; i++) {
        for(int j=0; j < 5; j++) {
            m_spin5x5[get_at(j, i, 5, 25)]->setDecimals(6);
            m_spin5x5[get_at(j, i, 5, 25)]->setMaximum(99.999999);
            m_spin5x5[get_at(j, i, 5, 25)]->setMinimum(-99.999999);
            m_spin5x5[get_at(j, i, 5, 25)]->setMinimumSize(QSize{90,24});
        }
    }

    for(int y=0; y < 5; y++) {
        for(int x = 0 ; x < 5; x++) {
            m_5x5layout[y]->addWidget(m_spin5x5[get_at(x, y, 5, 25)]);
            m_5x5layout[y]->setSizeConstraint(QLayout::SetDefaultConstraint);
        }
        ui->horizontalLayout_5x5->addLayout(m_5x5layout[y]);
    }

}

MainWindow::~MainWindow()
{
    delete ui;
    if (pCustKernel)
        delete pCustKernel;

}

void MainWindow::asyncInit()
{

    QObject::connect(ui->pushButton, SIGNAL(clicked()), this, SLOT(hClicked()));
    QObject::connect(ui->gaussianBlur, SIGNAL(clicked()), this, SLOT(hGaussianBlr()));
    QObject::connect(ui->intensiveSharper, SIGNAL(clicked()), this, SLOT(hIntensiveShrp()));
    QObject::connect(ui->sharper, SIGNAL(clicked()), this, SLOT(hSharper()));
    QObject::connect(ui->original, SIGNAL(clicked()), this, SLOT(hClickedOriginal()));
    QObject::connect(ui->grayscale, SIGNAL(clicked()), this, SLOT(hClickedGray()));
    QObject::connect(ui->identityBtn, SIGNAL(clicked()), this, SLOT(hClickedIdent()));
    QObject::connect(ui->spinBox, SIGNAL(valueChanged(int)), this, SLOT(hValChanged(int)));
    QObject::connect(ui->custKern, SIGNAL(clicked(bool)), this, SLOT(hCustKern3x3()));
    QObject::connect(ui->cust5x5, SIGNAL(clicked(bool)), this , SLOT(hCustKern5x5()));
    QObject::connect(ui->mtEnabled, SIGNAL(checkStateChanged(Qt::CheckState)), this , SLOT(hEnableMT(Qt::CheckState)));
    QObject::connect(ui->gpuAccelEn, SIGNAL(checkStateChanged(Qt::CheckState)), this , SLOT(hEnableGPU(Qt::CheckState)));
    //QObject::connect(&this->m_warmer, SIGNAL(timeout()), this, SLOT(hTimeout()));
    connect3x3();
    connect5x5();
//    m_warmer.start();

}


void MainWindow::convolveNxN(const QImage &qimg, eConvType type)
{
    int x, y;
    int total_len = m_rgbdata.size();
//    total_len = qimg.width() * qimg.height() ;


    if (type != eConvType::Custom5x5) {
        while (m_val-- >= 0)
        for(y=0; y < qimg.height()-2 ; y++ ) {
            for(x=0; x < qimg.width()-2; x++)
            {
                FRGB newpix;
                for(int i=0; i < TROW; i++) {
                    for(int j=0; j < TCOL; j++) {
                        m_rgbctx.rgbchans[i*TROW+j] = m_rgbdata.at(get_at(x+j,y+i, qimg.width(),total_len));
                        m_rgbctx.fRGB[i*TROW +j].rgb[0] = ((m_rgbctx.rgbchans[i*1 +j] >> 16) & LSBYTE);// / 255.0f;
                        m_rgbctx.fRGB[i*TROW +j].rgb[1] = ((m_rgbctx.rgbchans[i*1 +j] >> 8) & LSBYTE);// / 255.0f;
                        m_rgbctx.fRGB[i*TROW +j].rgb[2] = (m_rgbctx.rgbchans[i*1 +j] & LSBYTE);// / 255.0f;
                    }
                }
                switch (type) {
                case eConvType::GaussianBlur:
                    newpix= gaussianConv * m_rgbctx.fRGB;
                    break;
                case eConvType::Sharper:
                    newpix = sharpConv * m_rgbctx.fRGB;
                    break;
                case eConvType::IntensivSharper:
                    newpix = iSharpConv * m_rgbctx.fRGB;
                    break;
                case eConvType::Identity:
                    newpix = identConv * m_rgbctx.fRGB;
                    break;
                case eConvType::Custom3x3:
                    newpix = *pCustKernel * m_rgbctx.fRGB;
                    break;
                default:
                    break;
                }
                QRgb t = qRgba(newpix.rgb[0] , newpix.rgb[1], newpix.rgb[2] ,
                               qAlpha(m_rgbctx.rgbchans[4]));
                m_rgbdata[get_at(x,y, qimg.width(), total_len)] = t ;

            }
        }
    } else {
        while (m_val-- >= 0)
            for(y=0; y < qimg.height()-2 ; y++ ) {
                for(x=0; x < qimg.width()-2; x++)
                {
                    FRGB newpix;

                    for(int i=0; i < TROW2; i++) {
                        for(int j=0; j < TCOL2; j++) {
                            m_rgbctx.rgbchans2[i*TROW2+j] = m_rgbdata.at(get_at(x+j,y+i, qimg.width(),total_len));
                            m_rgbctx.fRGB2[i*TROW2 +j].rgb[0] = ((m_rgbctx.rgbchans2[i*1 +j] >> 16) & LSBYTE);// / 255.0f;
                            m_rgbctx.fRGB2[i*TROW2 +j].rgb[1] = ((m_rgbctx.rgbchans2[i*1 +j] >> 8) & LSBYTE);// / 255.0f;
                            m_rgbctx.fRGB2[i*TROW2 +j].rgb[2] = (m_rgbctx.rgbchans2[i*1 +j] & LSBYTE);// / 255.0f;
                        }
                    }
                    newpix = *pCustKernel * m_rgbctx.fRGB2;

                    QRgb t = qRgba(newpix.rgb[0] , newpix.rgb[1], newpix.rgb[2] ,
                                   qAlpha(m_rgbctx.rgbchans2[4]));
                    m_rgbdata[get_at(x,y, qimg.width(), total_len)] = t ;
                }
            }
    }

    QImage im2{qimg.width(), qimg.height(), QImage::Format_RGB32};
    QPixmap pm2 {qimg.width(), qimg.height()};

    for(int i=0; i < qimg.height(); i++)
        for(int j=0; j < qimg.width(); j++){
            im2.setPixel(j, i, m_rgbdata.at(get_at(j,i, qimg.width(), total_len)));
        }
    pm2.convertFromImage(im2);
    m_imgScreen.setPixmap(pm2);
    m_val = ui->spinBox->value();

}

//this will use 4 threads to convolv on piece
//!TODO
//! x0,y0       w/2
//! |------------|-------------|
//! |------------|-------------|
//! |------------h/2-----------|
//! |------------|-------------|
//! |------------|-------------|
//! |------------|-------------|
//!
void MainWindow::convolveNxNWorker(const QImage &qimg, eConvType type, int w, int h)
{

    p_helpers[0] = new worker_helper{0, 0, w/2,h/2 , m_val, type, this, m_rgbctx}; //top left
    p_helpers[1] = new worker_helper{w/2, 0 , w, h/2 ,m_val, type, this, m_rgbctx}; // top righ
    p_helpers[2] = new worker_helper{0, h/2, w/2, h ,m_val, type, this, m_rgbctx}; // bottom lrft
    p_helpers[3] = new worker_helper{w/2,h/2, w,h , m_val,type, this, m_rgbctx}; //bottom right


    for(int i=0; i < 4; i++) {
        p_helpers[i]->doWork();
    }
    for(int i=0; i < 4; i++)
        p_helpers[i]->m_worker.join();


    for(int i=0; i < 4; i++)
        delete p_helpers[i];

    QImage im2{qimg.width(), qimg.height(), QImage::Format_RGB32};
    QPixmap pm2 {qimg.width(), qimg.height()};
    auto total_len = qimg.height()*qimg.width();
    for(int i=0; i < qimg.height(); i++)
        for(int j=0; j < qimg.width(); j++){
            im2.setPixel(j, i, m_rgbdata.at(get_at(j,i, qimg.width(), total_len)));
        }
    pm2.convertFromImage(im2);
    m_imgScreen.setPixmap(pm2);
    m_val = ui->spinBox->value();
}

/// TODO: add the other filters
/// \brief MainWindow::convolveNxNAccel
/// accelerated via OPENCL
/// \param image of Qt image
/// \param type of filter
///
void MainWindow::convolveNxNAccel(const QImage &gimg, eConvType type)
{
    //    total_len = qimg.width() * qimg.height() ;
    if (type != eConvType::Custom5x5) {
        switch (type) {
        case eConvType::GaussianBlur:
            while (m_val-- >= 0)
            {
                m_gpukern.conv3x3(m_rgbdata.data(), 0,0, gimg.width(), gimg.height(), _blur);
            }
            break;
        case eConvType::Sharper: {
            while (m_val-- >= 0)
            {
                m_gpukern.conv3x3(m_rgbdata.data(), 0,0, gimg.width(), gimg.height(), _sharp);
            }
            break;
        }
        case eConvType::IntensivSharper: {
            while (m_val-- >= 0)
            {
                m_gpukern.conv3x3(m_rgbdata.data(), 0,0, gimg.width(), gimg.height(), _topsobel );
            }
            break;
        }
        case eConvType::Identity: {
            while (m_val-- >= 0)
            {
                m_gpukern.conv3x3(m_rgbdata.data(), 0,0, gimg.width(), gimg.height(), _ident );
            }
            break;
        }

        default:
            break;
        }

    }

    QImage im2{gimg.width(), gimg.height(), QImage::Format_RGB32};
    QPixmap pm2 {gimg.width(), gimg.height()};
    auto total_len = gimg.height()*gimg.width();
    for(int i=0; i < gimg.height(); i++)
        for(int j=0; j < gimg.width(); j++){
            im2.setPixel(j, i, m_rgbdata.at(get_at(j,i, gimg.width(), total_len)));
        }
    pm2.convertFromImage(im2);
    m_imgScreen.setPixmap(pm2);

    m_gpuAccel = true;
    m_val = ui->spinBox->value();

}

void MainWindow::to_gray(const QPixmap &ref, std::vector<unsigned int> & data)
{
    //0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.

    int x, y;
    int total_len ;
    QImage qimg = ref.toImage();
    total_len = qimg.width() * qimg.height() ;
    data.reserve(total_len);
    for(int i=0; i < qimg.height(); i++)
        for(int j=0; j < qimg.width(); j++)
            data.push_back(qimg.pixel(j, i));

    for(y=0; y < qimg.height() ; y++ ) {
        for(x=0; x < qimg.width(); x++) {
            Bits<unsigned int> f1 = {.val = data.at(get_at(x,y, qimg.width(), total_len))};
            float res = 0.299f * f1.data[0]   + 0.587f * f1.data[1]  +  0.114f * f1.data[2]  ;
            data[get_at(x,y, qimg.width(), total_len)] = clampnf<unsigned int>((unsigned int)res , 0, 0x00fffffful);
        }
    }

    QImage im2{qimg.width(), qimg.height(), QImage::Format_RGB32};
    QPixmap pm2 {qimg.width(), qimg.height()};

    for(int i=0; i < qimg.height(); i++)
        for(int j=0; j < qimg.width(); j++){
            im2.setPixel(j, i, data.at(get_at(j,i, qimg.width(), total_len)));
        }
    pm2.convertFromImage(im2);
    m_imgScreen.setPixmap(pm2);
}



void MainWindow::hClicked()
{
    QString fname = QFileDialog::getOpenFileName(this,tr("Open Image"), "/home/ilian/Downloads", tr("Image Files (*.png *.jpg *.bmp)"));
    m_currentPixmap = QPixmap{fname};
    m_imgScreen.setPixmap(m_currentPixmap);
    m_imgScreen.resize(m_imgScreen.pixmap().size());
//    ui->scrollArea->resize(m_imgScreen.size());
//        scrollArea->setWidgetResizable(fitToWindow);
    m_currImg = m_currentPixmap.toImage();
    m_rgbdata.clear();
    m_rgbdata.reserve(m_currImg.width() * m_currImg.width());
    for(int i=0; i < m_currImg.height(); i++) {
        for(int j=0; j < m_currImg.width(); j++){
            m_rgbdata.push_back(m_currImg.pixel(j,i));
        }
    }

    ui->scrollArea->setWidgetResizable(true);
    ui->pushButton->setEnabled(true);
    ui->gaussianBlur->setEnabled(true);
    ui->intensiveSharper->setEnabled(true);
    ui->sharper->setEnabled(true);
    ui->intensiveSharper->setEnabled(true);
    ui->grayscale->setEnabled(true);
    ui->custKern->setEnabled(true);
    ui->identityBtn->setEnabled(true);
    ui->original->setEnabled(true);


}

void MainWindow::hGaussianBlr()
{
    if (!m_mtenabled && !m_gpuAccel) {
        CHECKTIME (
        convolveNxN(m_currImg, eConvType::GaussianBlur);
            )
    } else if (m_mtenabled){
        CHECKTIME (
            convolveNxNWorker(m_currImg, eConvType::GaussianBlur, m_currImg.width(), m_currImg.height());
            )
    } else if (m_gpuAccel) {
        CHECKTIME (
            convolveNxNAccel(m_currImg, eConvType::GaussianBlur);
            )
    }
}


void MainWindow::hIntensiveShrp()
{
    convolveNxN(m_currImg, eConvType::IntensivSharper);
}

void MainWindow::hSharper()
{
    convolveNxN(m_currImg, eConvType::Sharper);
}

void MainWindow::hClickedIdent()
{
    convolveNxN(m_currImg, eConvType::Identity);
}

void MainWindow::hClickedOriginal()
{
    m_imgScreen.setPixmap(m_currentPixmap);
    m_imgScreen.resize(m_imgScreen.pixmap().size());
    copydata();
}

void MainWindow::hValChanged(int v)
{
    m_val = v;
    qDebug() << m_val;
}

void MainWindow::hCustKern3x3()
{
    convolveNxN(m_currImg, eConvType::Custom3x3);
}

void MainWindow::hCustKern5x5()
{
    convolveNxN(m_currImg, eConvType::Custom5x5);
}

void MainWindow::hEnableMT(Qt::CheckState state)
{
    switch (state) {
    case Qt::CheckState::Checked:
        m_mtenabled = true;
        break;
    case Qt::CheckState::Unchecked:
    default :
        m_mtenabled = false;
        break;
    }
}

void MainWindow::hEnableGPU(Qt::CheckState state)
{
    switch(state) {
    case Qt::CheckState::Checked:
        m_gpuAccel = true;
        break;
    case Qt::CheckState::Unchecked:
    default:
        m_gpuAccel = false;
        break;
    }
}

void MainWindow::hClickedGray()
{
    to_gray(m_currentPixmap, m_rgbdata);
}

void MainWindow::connect3x3()
{
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            m_3x3slots[get_at(j,i, 3, 9)] = [=](double f)
            {
                qDebug() << f << "|" << j << "|" <<i;
                pCustKernel->val3x3[get_at(j,i,3,9)] = f;
//                convolveNxN(m_currImg, eConvType::Custom3x3);
            };
        }
    }

    for(int i=0; i < 3; i++) {
        for (int j=0; j < 3; j++) {
            connect(m_spin3x3[get_at(j,i,3,9)], &QDoubleSpinBox::valueChanged, m_3x3slots[get_at(j,i,3,9)]);
        }
    }

}

void MainWindow::connect5x5()
{    
    for(int i=0; i < 5; i++) {
        for(int j=0; j < 5; j++) {
            m_5x5slots[get_at(j,i, 5, 25)] = [=](double f)
            {
                qDebug() << f << "|" << j << "|" <<i;
                pCustKernel->val5x5[get_at(j,i,5,25)] = f;
             //   convolveNxN(m_currImg, eConvType::Custom5x5);

            };
        }
    }

    for(int i=0; i < 5; i++) {
        for (int j=0; j < 5; j++) {
            connect(m_spin5x5[get_at(j,i,5,25)], &QDoubleSpinBox::valueChanged, m_5x5slots[get_at(j,i,5,25)]);
        //    convolveNxN(m_currImg, eConvType::Custom3x3);
        }
    }
}

void MainWindow::copydata()
{
    m_currImg = m_currentPixmap.toImage();
    m_rgbdata.clear();
    m_rgbdata.reserve(m_currImg.width() * m_currImg.width());
    for(int i=0; i < m_currImg.height(); i++) {
        for(int j=0; j < m_currImg.width(); j++){
            m_rgbdata.push_back(m_currImg.pixel(j,i));
        }
    }
}

void MainWindow::hTimeout()
{
    m_gpukern.warmer();
}
