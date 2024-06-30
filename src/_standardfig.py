import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl
import numpy as np
import scicomap as sc

class ImageParameters:
    """
    初始化 ImageParameters 类，根据传入的 layout 参数设置图像的各种属性。

    参数:
    layout: str
        布局类型，可选值包括：
        'single_column': 单栏图形
        'double_column' or 'double_columns': 双栏图形
        'full_page': 全页图形

    属性:
    fig_width: float
        图像的宽度。
    title_fontsize: int
        标题字体大小。
    label_fontsize: int
        标签字体大小。
    label_tick_size: int
        刻度字体大小。
    text_fontsize: int
        文本字体大小。
    """

    def __init__(self, layout):
        # 设置子图的默认标题
        self.subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', 
                         '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']

        # 根据不同的布局类型设置图像参数
        if layout == 'single_column':
            self.fig_width = 3.5
            self.title_fontsize = 9
            self.label_fontsize = 8
            self.label_tick_size = 8
            self.text_fontsize = 8

        elif layout in ['double_column', 'double_columns']:
            self.fig_width = 7
            self.title_fontsize = 12
            self.label_fontsize = 10
            self.label_tick_size = 10
            self.text_fontsize = 8

        elif layout == 'full_page':
            self.fig_width = 8.5
            self.title_fontsize = 14
            self.label_fontsize = 12
            self.label_tick_size = 12

class StandardFigure:
    """
    创建并管理标准化图形的类，支持2D和3D视图。

    方法:
    set_base_para: 设置基本参数。
    auto_set_fontname: 自动设置全局字体。
    compute_fig_height: 计算图像的总高度。
    create_figure: 创建图像并设置图像对象。
    """

    def set_base_params(self, layout='single_column', rows=1, columns=1, weight_height_ratio=1, projection3d=False):
        """
        设置图像的基本参数。

        参数:
        layout: str, 图像的布局方式，默认为 'single_column'。
        rows: int, 图像的行数，默认为 1。
        columns: int, 图像的列数，默认为 1。
        weight_height_ratio: float, 图像的宽高比，默认为 1。
        projection3d: bool, 是否为3D图，默认为 False。
        """
        self.rows, self.columns = rows, columns
        self.weight_height_ratio = weight_height_ratio
        self.projection3d = projection3d
        self.params = ImageParameters(layout=layout)  # 使用更正式的变量名
        self.auto_set_fontname()
        self.fig_height = self.compute_fig_height()  # 计算图像的总高度
        self.num_subplots = self.rows * self.columns  # 计算子图的总数量

    def auto_set_fontname(self):
        """
        自动设置Matplotlib的全局字体参数以支持多语言。
        """
        import matplotlib  # 确保在函数中导入matplotlib
        matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 用黑体显示中文
        matplotlib.rcParams['mathtext.default'] = 'it'
        matplotlib.rcParams['mathtext.fallback'] = 'cm'
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['axes.unicode_minus'] = False

    def compute_fig_height(self):
        """
        计算图像的总高度，基于子图的宽度和宽高比。

        返回:
        float, 计算得到的图像总高度。
        """
        subfig_width = self.params.fig_width / self.columns  # 每个子图的宽度
        subfig_height = subfig_width * self.weight_height_ratio  # 每个子图的高度
        fig_height = subfig_height * self.rows  # 图像的总高度
        return fig_height

    def create_figure(self):
        """
        根据设置创建2D或3D图像。

        返回:
        fig, axs: Matplotlib Figure和Axes对象。
        """
        if not self.projection3d:
            self.fig, self.axs = plt.subplots(self.rows, self.columns, 
                                              figsize=(self.params.fig_width, self.fig_height), 
                                              dpi=300)
        else:
            self.fig, self.axs = plt.subplots(self.rows, self.columns, 
                                              figsize=(self.params.fig_width, self.fig_height), 
                                              subplot_kw={'projection': '3d'},
                                              dpi=100)
    
    def get_subfig(self, row=None, column=None, index=None):

        if self.num_subplots == 1:
            self.ax = self.axs
            return None

        if index is not None:
            if self.rows == 1:
                self.ax = self.axs[index-1]
                return None
            if self.rows > 1:
                if self.columns == 1:
                    self.ax = self.axs[index-1]
                else:
                    row = (index-1) // self.columns
                    column = index % self.columns
                    self.ax = self.axs[row, column-1]
                return None
            
        if row and column is not None:
            if self.rows==1 and row==1:
                self.ax = self.axs[column-1]
                return None  
            if self.rows > 1:
                self.ax = self.axs[row-1, column-1]
                return None          
    
    @staticmethod
    def create_two_slope_norm(data, equal_lengths=True):
        """
        创建一个具有两个斜率的归一化对象，用于颜色映射。

        参数:
        data: 数组，输入数据，用于确定归一化范围。
        equal_lengths: 布尔值，是否在正负范围内使用相同的长度。

        返回值:
        norm: 具有两个斜率的归一化对象。
        """
        if equal_lengths:
            # 如果正负范围使用相同长度，则取数据绝对值的最大值作为范围
            data_max = np.max(abs(data))
            vmin = -data_max
            vcenter = 0
            vmax = data_max
        else:
            # 如果不使用相同长度，则取数据的最小值和最大值作为范围
            vmin = np.min(data)
            vmax = np.max(data)
            # 判断数据是否跨越零点
            if vmin * vmax < 0:
                vcenter = 0
            else:
                vcenter = (vmax + vmin) / 2

        # 创建具有两个斜率的归一化对象
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        return norm
    
    @staticmethod
    def get_scico_colormap(map_name):
        """
        获取指定名称的 Scico 分散颜色映射，并转换为 Matplotlib 颜色映射。

        参数:
        map_name: 字符串,Scico 分散颜色映射的名称。

        返回值:
        cmap: Matplotlib 颜色映射对象。
        """
        # 创建一个 Scico 分散颜色映射对象
        scico_diverging_map = sc.ScicoDiverging(cmap=map_name)
        # 获取对应的 Matplotlib 颜色映射
        mpl_colormap = scico_diverging_map.get_mpl_color_map()
        return mpl_colormap
    
    def create_colorbar(self, image, aspect_ratio=14, padding_fraction=1, **kwargs):
        """
        为图像添加一个垂直的颜色条。

        参数:
        image: 图像对象，需要添加颜色条的图像。
        aspect_ratio: 颜色条的纵横比，默认为 14。
        padding_fraction: 颜色条与图像之间的间距比例，默认为 1。
        **kwargs: 传递给 colorbar 的其他参数。

        返回值:
        colorbar: 添加到图像的颜色条对象。
        """
        # bounds = [self.norm.vmin, self.norm.vcenter, self.norm.vmax]
        # mappable = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        # image=mappable
        # 创建一个与图像关联的可分隔轴对象
        divider = make_axes_locatable(image.axes)
        # 计算颜色条的宽度
        colorbar_width = axes_size.AxesY(image.axes, aspect=1./aspect_ratio)
        # 计算颜色条与图像之间的间距
        padding = axes_size.Fraction(padding_fraction, colorbar_width)
        # 获取当前的坐标轴
        current_axes = plt.gca()
        # 在图像右侧添加一个新的轴用于放置颜色条
        colorbar_axes = divider.append_axes("right", size=colorbar_width, pad=padding)
        # 设置当前轴为之前的轴
        plt.sca(current_axes)
        # 为图像添加颜色条
        colorbar = image.axes.figure.colorbar(image, cax=colorbar_axes, **kwargs)
        # colorbar = self.fig.colorbar(mappable, cax=colorbar_axes, ticks=bounds)

        return colorbar

    def set_imshow_params(self, data, cmap_name='seismic', equal_length_norm=True, interpolation='lanczos', colorbar=True, invert_yaxis=True):
        """
        设置用于显示图像数据的参数。

        参数:
        data: array_like, 显示的数据。
        cmap_name: str, 使用的颜色映射，默认为 'seismic'。
        equal_length_norm: bool, 是否使用等长度归一化，默认为 True。
        interpolation: str, 图像的插值方法，默认为 'lanczos'。
        colorbar: bool, 是否显示颜色条，默认为 True。
        invert_yaxis: bool, 是否翻转y轴, 默认为 True。
        """
        self.data = data
        self.cmap = self.get_scico_colormap(cmap_name)  # 获取颜色映射
        self.interpolation = interpolation
        self.colorbar = colorbar
        self.equal_length_norm = equal_length_norm
        self.invert_yaxis = invert_yaxis

    def show_data(self):
        """
        显示数据。根据设定的参数使用 matplotlib 的 imshow 方法渲染数据。
        """
        # 创建双斜率归一化对象
        self.norm = self.create_two_slope_norm(self.data, equal_lengths=self.equal_length_norm)
        # 使用 imshow 方法显示数据
        data_map = self.ax.imshow(self.data, norm=self.norm, cmap=self.cmap, interpolation=self.interpolation)
        # 如果需要，添加颜色条
        if self.colorbar:
            self.create_colorbar(data_map)
        if self.invert_yaxis:
            self.ax.invert_yaxis()

    def set_ticks_params(self, x_tick_label_step, y_tick_label_step,
                         x_decimal=0, y_decimal=0,
                         max_bins_x=None, max_bins_y=None, 
                         x_num=None, y_num=None, x_tick_interval=None, y_tick_interval=None,
                         custom_x_ticks=None, custom_y_ticks=None):
        """
        设置x轴和y轴的刻度参数。

        参数:
        x_tick_label_step: float, x轴每个刻度的标签间隔。
        y_tick_label_step: float, y轴每个刻度的标签间隔。
        x_decimal: int, x轴标签的小数位数。
        y_decimal: int, y轴标签的小数位数。
        max_bins_x: int, x轴的最大刻度数量。
        max_bins_y: int, y轴的最大刻度数量。
        x_num: int, x轴的数据点数量。
        y_num: int, y轴的数据点数量。
        x_tick_interval: int, x轴的刻度间隔。
        y_tick_interval: int, y轴的刻度间隔。
        custom_x_ticks: list, 自定义x轴的刻度位置。
        custom_y_ticks: list, 自定义y轴的刻度位置。
        """
        self.x_tick_type = None
        self.y_tick_type = None
        
        self.x_decimal = x_decimal
        self.y_decimal = y_decimal

        # 根据自定义刻度位置直接绘制刻度
        if custom_x_ticks is not None:
            self.x_tick_type = 'custom'
            self.x_ticks = custom_x_ticks
            self.x_tick_labels = [f"{x_tick_label_step*element:.{x_decimal}f}" for element in custom_x_ticks]
        if custom_y_ticks is not None:
            self.y_tick_type = 'custom'
            self.y_ticks = custom_y_ticks
            self.y_tick_labels = [f"{y_tick_label_step*element:.{y_decimal}f}" for element in custom_y_ticks]

        # 根据最大刻度数量配置刻度
        if max_bins_x is not None and self.x_tick_type is None:
            self.x_tick_type = 'max_bins'
            self.max_bins_x = max_bins_x
        if max_bins_y is not None and self.y_tick_type is None:
            self.y_tick_type = 'max_bins'
            self.max_bins_y = max_bins_y

        # 通过固定的刻度间隔配置刻度
        if x_num and x_tick_interval is not None and self.x_tick_type is None:
            self.x_tick_type = 'interval'
            self.x_ticks = list(range(0, x_num, x_tick_interval))
            self.x_tick_labels = [f"{x_tick_label_step*element:.{x_decimal}f}" for element in self.x_ticks]
        if y_num and y_tick_interval is not None and self.y_tick_type is None:
            self.y_tick_type = 'interval'
            self.y_ticks = list(range(0, y_num, y_tick_interval))
            self.y_tick_labels = [f"{y_tick_label_step*element:.{y_decimal}f}" for element in self.y_ticks]

    def add_axis_ticks(self):
        """
        根据先前设置的参数添加轴的刻度。
        """
        import matplotlib.ticker as ticker  # 确保在方法中导入所需模块

        # 根据不同的刻度类型配置轴
        if self.x_tick_type == 'max_bins':
            self.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=self.max_bins_x))
        else:
            self.ax.set_xticks(self.x_ticks)
            self.ax.set_xticklabels(self.x_tick_labels)
        
        if self.y_tick_type == 'max_bins':
            self.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=self.max_bins_y))
        else:
            self.ax.set_yticks(self.y_ticks)
            self.ax.set_yticklabels(self.y_tick_labels)

        # 配置标签的数字格式
        from matplotlib.ticker import FormatStrFormatter
        self.ax.xaxis.set_major_formatter(FormatStrFormatter(f"%{'.%df' % self.x_decimal}" if self.x_decimal > 0 else "%d"))
        self.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%{'.%df' % self.y_decimal}" if self.y_decimal > 0 else "%d"))

    def set_label_and_title_params(self, x_label=None, y_label=None, z_label=None, subtitle=None):
        """
        设置图像的标签和标题参数。

        参数:
        x_label: str, x轴标签, 默认为 'x'。
        y_label: str, y轴标签, 默认为 'y'。
        z_label: str, y轴标签, 默认为 'z'。
        subtitle: str, 图像的子标题。
        """
        # 设置x轴标签，默认为 'x'
        self.x_label = x_label if x_label is not None else 'x'
        # 设置y轴标签，默认为 'y'
        self.y_label = y_label if y_label is not None else 'y'
        # 设置z轴标签，默认为 'z'
        self.z_label = z_label if z_label is not None else 'z'
        # 设置子标题
        self.subtitle = subtitle

    def add_labels_and_title(self):
        """
        在图像上添加设置的标签和标题。
        """
        # 设置标题，如果提供了子标题
        if self.subtitle:
            self.ax.set_title(self.subtitle, fontsize=self.params.title_fontsize)
        # 设置x轴和y轴标签
        self.ax.set_xlabel(self.x_label, fontsize=self.params.label_fontsize)
        self.ax.set_ylabel(self.y_label, fontsize=self.params.label_fontsize)
        if self.projection3d:
            self.ax.set_zlabel(self.z_label, fontsize=self.params.label_fontsize, rotation=90, verticalalignment='center',  labelpad=0.1)
        # 设置刻度标签的字体大小
        self.ax.tick_params(labelsize=self.params.label_tick_size)

    def set_batch_norm(self, vmin, vcenter, vmax):
        """"
        设置数据的双斜率归一化参数。

        参数:
        vmin: float, 归一化的最小值。
        vcenter: float, 归一化的中心值。
        vmax: float, 归一化的最大值。
        """
        self.norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    def batch_show_data(self):
        """
        批量显示数据，每个子图显示一个数据集。
        """
        for i in range(self.num_subplots):
            print(f"Processing subplot {i+1}")
            try:
                self.get_subfig(index=i+1)
                data_map = self.ax.imshow(self.data[i], norm=self.norm, cmap=self.cmap, interpolation=self.interpolation)
                if self.colorbar:
                    self.create_colorbar(data_map)
                if self.invert_yaxis:
                    self.ax.invert_yaxis()
            except Exception as e:
                print(f"Error processing subplot {i+1}: {e}")
                break

    def batch_set_labels_and_titles(self):
        """
        批量设置每个子图的标签和标题。
        """
        if self.subtitle is None:
            self.subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', 
                             '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
        for i in range(self.num_subplots):
            self.get_subfig(index=i+1)
            self.ax.set_title(self.subtitle[i], fontsize=self.params.title_fontsize)
            self.ax.set_xlabel(self.x_label, fontsize=self.params.label_fontsize)
            self.ax.set_ylabel(self.y_label, fontsize=self.params.label_fontsize)
            self.ax.tick_params(labelsize=self.params.label_tick_size)
            
    def batch_add_axis_ticks(self):
        """
        批量添加坐标轴刻度。
        """
        for i in range(self.num_subplots):
            self.get_subfig(index=i+1)
            self.add_axis_ticks()

if __name__ == "__main__":

    from _gendata import gendata
    data = gendata()   # 生成测试数据

    layout = 'double_column'  # 设置图形布局为双栏
    rows = 2                  # 设置子图行数为2
    columns = 3               # 设置子图列数为3
    projection3d = False      # 设置是否为3D图形，这里为2D

    data0 = data[0].T  # 转置第一个数据集，通常用于调整数据的行列关系
    # 计算图像的宽高比，通常用于保证图像的显示比例合理
    weight_height_ratio = data0.shape[1]/data0.shape[0]

    fig1 = StandardFigure()   # 创建一个标准图形实例

    # 设置基本参数，初始化图形布局和尺寸
    fig1.set_base_params(layout=layout, rows=rows, columns=columns, 
                        weight_height_ratio=weight_height_ratio, 
                        projection3d=projection3d)
    fig1.create_figure()      # 创建图形

    # index = 6                 # 设置索引，可能用于选择特定的子图
    # fig1.get_subfig(index=index)  # 获取指定索引的子图，此行代码似乎未完成实现

    # data0 = data[0].T         # 重新转置数据，看起来是重复的步骤
    cmap_name = 'seismic'          # 设置颜色映射为地震颜色
    equal_length_norm=True    # 设置是否使用等长度的归一化
    interpolation='none'      # 设置插值方式为无，保持数据像素的原始显示
    invert_yaxis = True

    # 设置用于显示图像的参数
    fig1.set_imshow_params(data=data, cmap_name=cmap_name, 
                           equal_length_norm=equal_length_norm,
                           interpolation=interpolation)
    fig1.set_batch_norm(vmin=-100, vcenter=0, vmax=100)  # 设置归一化参数
    fig1.batch_show_data()  # 批量显示数据

    x_label = 'x(km)'        # 设置x轴标签
    y_label = '深度/100'     # 设置y轴标签
    subtitle = None          # 未设置子标题（使用默认子标题）
    fig1.set_label_and_title_params(x_label=x_label, y_label=y_label, subtitle=subtitle)
    fig1.batch_set_labels_and_titles()  # 批量设置子图的标签和标题

    x_tick_label_step, y_tick_label_step = 1.0, 2.0  # 设置刻度标签步长
    max_bins_x, max_bins_y = 5, 4                   # 设置最大刻度数
    x_decimal = 1                                   # 设置x轴刻度的小数位数
    y_decimal = 0                                   # 设置y轴刻度的小数位数

    (y_num, x_num) = data[1].shape                  # 获取第二个数据集的形状
    x_tick_interval, y_tick_interval = 10, 10       # 设置刻度间隔

    # 设置刻度参数
    fig1.set_ticks_params(x_tick_label_step=x_tick_label_step, y_tick_label_step=y_tick_label_step,
                        x_decimal=x_decimal, y_decimal=y_decimal,
                        x_num=x_num, y_num=y_num, x_tick_interval=x_tick_interval, y_tick_interval=y_tick_interval)
    fig1.batch_add_axis_ticks()  # 批量添加刻度

    fig1.fig.tight_layout()      # 自动调整布局，避免内容重叠

    plt.savefig('test.tiff')     # 保存图像到文件
    plt.show()                   # 显示图像