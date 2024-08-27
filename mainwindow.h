#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "defs.h"
#include "convaccel.h"
#include <QMainWindow>
#include <QPixmap>
#include <QLabel>
#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QThread>

#include <functional>
#include <thread>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE



enum class eConvType
{
    GaussianBlur ,
    Sharper,
    IntensivSharper,
    Identity,
    Original,
    Custom3x3,
    Custom5x5,
    COUNT
};


struct kernel_t;


struct Warmer : public QThread
{    Q_OBJECT

public:
    Warmer(gpu_kernel* kernref) : m_kernref{kernref}
    {

    }
    void run() override {
        bool done = false;
        for(;;) {
            m_kernref->warmer();
            this->sleep(2000);
        }
        /* ... here is the expensive or blocking operation ... */
        emit resultReady(done);
    }
signals:
    void resultReady(bool);
private:

    gpu_kernel* m_kernref;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


    void asyncInit();
private:
    Ui::MainWindow *ui;

    void convolveNxN(const QImage &qimg, eConvType type);
    void convolveNxNWorker(const QImage &qimg, eConvType type, int w, int h);
    void convolveNxNAccel(const QImage& gimg, eConvType type);
    void hConvMT();
    void to_gray(const QPixmap& ref, std::vector<unsigned int>& );
    void connect3x3();
    void connect5x5();
    void copydata();

private slots:
    void hClicked();
    void hGaussianBlr();
    void hIntensiveShrp();
    void hSharper();
    void hClickedIdent();
    void hClickedGray();
    void hClickedOriginal();
    void hValChanged(int);
    void hCustKern3x3();
    void hCustKern5x5();
    void hEnableMT(Qt::CheckState state);
    void hEnableGPU(Qt::CheckState state);
    void hTimeout();

private:
    int m_val;
    bool m_mtenabled;
    bool m_gpuAccel;
    std::vector<unsigned int> m_rgbdata;
    QImage m_currImg;
    kernel_t* pCustKernel;
    QString m_filename;
    QPixmap m_currentPixmap;
    QLabel m_imgScreen;

    QVBoxLayout* m_3x3layout[3];
    QVBoxLayout* m_5x5layout[5];

    QDoubleSpinBox* m_spin3x3[9];
    QDoubleSpinBox* m_spin5x5[25];

    std::function<void (double)> m_3x3slots[9];
    std::function<void (double)> m_5x5slots[25];

    struct rgbctx {
        QRgb rgbchans[9] = {0};
        FRGB fRGB[9] ;
        QRgb rgbchans2[25] = {0};
        FRGB fRGB2[25] ;
    } m_rgbctx;


    struct worker_helper{
        worker_helper(int x, int y, int w, int h, int count,
                      eConvType type,
                      MainWindow* parent, rgbctx& ctx):
            m_x{x}, m_y{y}, m_w{w}, m_h{h},m_count{count},
            m_type{type},
            p_Parent{parent}
        {
            memcpy(&m_localCtx, &ctx, sizeof(rgbctx));
        }

        void doWork();
        void doWorkAccel();
        void COW();
        void COW_BACK();
        int m_x, m_y, m_w, m_h, m_count;
        eConvType m_type;
        MainWindow* p_Parent;
        std::thread m_worker;
        struct rgbctx m_localCtx;

    };

    worker_helper* p_helpers[8];
    gpu_kernel m_gpukern;
    Warmer m_warmer;

};
#endif // MAINWINDOW_H
