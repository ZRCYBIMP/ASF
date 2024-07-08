import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from src._standardfig import StandardFigure
from src._gendata import gendata

class PlotCanvas(FigureCanvas):
    # def __init__(self, parent=None, width=5, height=4, dpi=100):
    #     # self.fig = Figure(figsize=(width, height), dpi=dpi)
    #     # self.axes = self.fig.add_subplot(111)
    #     # super().__init__(self.fig)
    #     # self.setParent(parent)

    def plot(self):
        # self.fig.clf()  # 清除旧图形
        # self.axes = self.fig.add_subplot(121)  # 重新添加子图
        data = gendata()
        sf = StandardFigure()
        sf.set_base_params(layout='double_column', rows=2, columns=3, weight_height_ratio=1, projection3d=False)
        sf.create_figure()
        sf.get_subfig(index=1)

        sf.set_imshow_params(data=data[0], cmap_name='seismic', equal_length_norm=True, interpolation='none', invert_yaxis=True)
        sf.show_data()
        self.draw()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt6 Matplotlib Example'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        
        m = PlotCanvas(self, width=5, height=4)
        m.move(0, 0)

        button = QPushButton('Plot', self)
        button.setToolTip('Click to plot data')
        button.resize(button.sizeHint())
        button.move(500, 0)
        button.clicked.connect(m.plot)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    
    sys.exit(app.exec())









































































































































































































































































































































































