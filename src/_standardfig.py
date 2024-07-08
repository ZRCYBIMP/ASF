import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl
import numpy as np
import scicomap as sc
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

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
    创建并管理标准化图形的类, 支持2D和3D视图。

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
                                              constrained_layout=True, dpi=300)
        else:
            self.fig, self.axs = plt.subplots(self.rows, self.columns, 
                                              figsize=(self.params.fig_width, self.fig_height), 
                                              subplot_kw={'projection': '3d'},
                                              constrained_layout=True, dpi=300)
    
    def get_subfig(self, row=None, column=None, index=None):
        """
        根据给定的行、列或索引获取子图的引用。

        参数:
        row (int): 子图所在的行。
        column (int): 子图所在的列。
        index (int): 子图的索引，适用于一维布局。

        返回:
        None, 但会设置 self.ax 为对应的子图。
        """
        if self.num_subplots == 1:
            self.ax = self.axs
            return

        if index is not None:
            # 计算行和列索引
            row_idx = (index - 1) // self.columns
            col_idx = (index - 1) % self.columns
            self.ax = self.axs[row_idx, col_idx] if self.rows > 1 and self.columns > 1 else self.axs[index - 1]
            return

        if row is not None and column is not None:
            self.ax = self.axs[row - 1, column - 1]
            return
        
    def create_two_slope_norm(self, data=None, equal_lengths=True,
                              vmin=None, vcenter=None, vmax=None):
        """
        创建具有两个斜率的归一化对象。此方法可以自动处理正负数值范围的归一化。

        参数:
        data (array-like): 输入数据，用于计算归一化的范围。
        equal_lengths (bool): 是否在正负范围内使用相同长度。
        vmin (float): 归一化的最小值。
        vcenter (float): 归一化的中心值。
        vmax (float): 归一化的最大值。

        返回:
        None, 但会设置 self.norm 为创建的归一化对象。
        """
        try:
            if data is not None:
                # 根据数据自动计算归一化范围
                if equal_lengths:
                    data_max = np.max(np.abs(data))
                    vmin = -data_max
                    vcenter = 0
                    vmax = data_max
                else:
                    vmin = np.min(data)
                    vmax = np.max(data)
                    if vmin * vmax < 0:  # 数据跨零点
                        vcenter = 0
                    else:
                        vcenter = (vmax + vmin) / 2

            if vmin is not None and vmax is not None:
                self.norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                raise ValueError("Minimum and maximum values are required for normalization.")
        except Exception as e:
            print(f"Error creating two slope normalization: {e}")


    def create_norm(self, data=None, equal_lengths=None, vmin=None, vmax=None):
        """
        根据提供的数据创建一个线性归一化对象。可选择是否使用等长的正负值范围。

        参数:
        data (array-like): 输入数据，用于计算归一化的范围。
        equal_lengths (bool): 是否在正负范围内使用相同长度。
        vmin (float): 归一化的最小值，如果不提供将根据数据计算。
        vmax (float): 归一化的最大值，如果不提供将根据数据计算。

        返回:
        None, 但会设置 self.norm 为创建的归一化对象。
        """
        try:
            if data is not None and equal_lengths is not None:
                # 根据数据自动计算归一化范围
                if equal_lengths:
                    data_max = np.max(np.abs(data))
                    vmin = -data_max
                    vmax = data_max
                else:
                    vmin = np.min(data)
                    vmax = np.max(data)

            if vmin is not None and vmax is not None:
                self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                raise ValueError("Minimum and maximum values are required for normalization.")
        except Exception as e:
            print(f"Error creating normalization: {e}")

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
    
    def create_colorbar(self, image, padding_fraction=1, **kwargs):
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
        # 创建一个与图像关联的可分隔轴对象
        divider = make_axes_locatable(image.axes)
        # 计算颜色条的宽度
        colorbar_width = axes_size.AxesY(image.axes, aspect=self.cbar_aspect)
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

    def set_imshow_params(self, data_list=None, data=None, cmap_name='seismic', equal_length_norm=True, interpolation='lanczos', invert_yaxis=True, colorbar=True):
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
        self.data_list = data_list
        self.cmap = self.get_scico_colormap(cmap_name)  # 获取颜色映射
        self.interpolation = interpolation
        self.equal_length_norm = equal_length_norm
        self.invert_yaxis = invert_yaxis
        self.colorbar = colorbar

    def show_data(self):
        """
        显示数据。根据设定的参数使用 matplotlib 的 imshow 方法渲染数据。
        """
        # 创建双斜率归一化对象
        # 使用 imshow 方法显示数据
        self.map = self.ax.imshow(self.data, norm=self.norm, cmap=self.cmap, interpolation=self.interpolation)
        if self.colorbar:
            self.add_colorbar()
        if invert_yaxis:
            self.ax.invert_yaxis()

    def set_ticks_params(self, x_tick_label_step, y_tick_label_step, z_tick_label_step=None,
                         x_decimal=0, y_decimal=0, z_decimal=0,
                         x_tick_interval=None, y_tick_interval=None, z_tick_interval=None,
                         custom_x_ticks=None, custom_y_ticks=None, custom_z_ticks=None):
        """
        设置x轴和y轴的刻度参数。

        参数:
        x_tick_label_step: float, x轴每个刻度的标签间隔。
        y_tick_label_step: float, y轴每个刻度的标签间隔。
        x_decimal: int, x轴标签的小数位数。
        y_decimal: int, y轴标签的小数位数
        x_num: int, x轴的数据点数量。
        y_num: int, y轴的数据点数量。
        x_tick_interval: int, x轴的刻度间隔。
        y_tick_interval: int, y轴的刻度间隔。
        custom_x_ticks: list, 自定义x轴的刻度位置。
        custom_y_ticks: list, 自定义y轴的刻度位置。
        """
        self.x_tick_type = None
        self.y_tick_type = None
        self.z_tick_type = None
        self.x_decimal = x_decimal
        self.y_decimal = y_decimal
        self.z_decimal = z_decimal

        if z_tick_label_step is not None:
            self.y_num, self.x_num, self.z_num = self.data.shape
        else:
            self.y_num, self.x_num = self.data.shape

        # 根据自定义刻度位置直接绘制刻度
        if custom_x_ticks is not None:
            self.x_tick_type = 'custom'
            self.x_ticks = custom_x_ticks
            self.x_tick_labels = [f"{x_tick_label_step*element:.{x_decimal}f}" for element in custom_x_ticks]
        if custom_y_ticks is not None:
            self.y_tick_type = 'custom'
            self.y_ticks = custom_y_ticks
            self.y_tick_labels = [f"{y_tick_label_step*element:.{y_decimal}f}" for element in custom_y_ticks]
        if custom_z_ticks is not None:
            self.z_tick_type = 'custom'
            self.z_ticks = custom_z_ticks
            self.z_tick_labels = [f"{z_tick_label_step*element:.{z_decimal}f}" for element in custom_z_ticks]

        # 通过固定的刻度间隔配置刻度
        if x_tick_interval is not None and self.x_tick_type is None:
            self.x_tick_type = 'interval'
            self.x_ticks = list(range(0, self.x_num, x_tick_interval))
            self.x_tick_labels = [f"{x_tick_label_step*element:.{x_decimal}f}" for element in self.x_ticks]
        if y_tick_interval is not None and self.y_tick_type is None:
            self.y_tick_type = 'interval'
            self.y_ticks = list(range(0, self.y_num, y_tick_interval))
            self.y_tick_labels = [f"{y_tick_label_step*element:.{y_decimal}f}" for element in self.y_ticks]
        if z_tick_interval is not None and self.z_tick_type is None:
            self.z_tick_type = 'interval'
            self.z_ticks = list(range(0, self.z_num, z_tick_interval))
            self.z_tick_labels = [f"{z_tick_label_step*element:.{z_decimal}f}" for element in self.z_ticks]

    def add_axis_ticks(self):
        """
        根据先前设置的参数添加轴的刻度。
        """
        # 根据不同的刻度类型配置轴
        self.ax.set_xticks(self.x_ticks)
        self.ax.set_xticklabels(self.x_tick_labels, fontsize=self.params.label_tick_size)
        self.ax.set_yticks(self.y_ticks)
        self.ax.set_yticklabels(self.y_tick_labels, fontsize=self.params.label_tick_size)

        # 配置标签的数字格式
        self.ax.xaxis.set_major_formatter(FormatStrFormatter(f"%{'.%df' % self.x_decimal}" if self.x_decimal > 0 else "%d"))
        self.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%{'.%df' % self.y_decimal}" if self.y_decimal > 0 else "%d"))
        if self.z_tick_type is not None:
            self.ax.set_zticks(self.z_ticks)
            self.ax.set_zticklabels(self.z_tick_labels, fontsize=self.params.label_tick_size)
            self.ax.zaxis.set_major_formatter(FormatStrFormatter(f"%{'.%df' % self.z_decimal}" if self.z_decimal > 0 else "%d"))

    def set_colorbar_ticks_params(self, cbar_name=None, cbar_ticks=None, cbar_decimal=0, cbar_aspect=14, orientation='vertical'):
        """
        设置颜色条的参数。

        参数:
        cbar_name (str): 颜色条的标签名称。
        cbar_ticks (list): 颜色条上的刻度值。
        cbar_decimal (int): 颜色条刻度标签的小数位数。

        返回:
        None
        """
        self.cbar_name = cbar_name
        self.cbar_ticks = cbar_ticks
        self.cbar_decimal = cbar_decimal
        self.cbar_aspect = cbar_aspect
        self.orientation = orientation
    
    def add_colorbar(self):
        """
        为图像添加颜色条。

        尝试使用自定义方法创建颜色条。如果失败，则使用matplotlib的默认方法。

        返回:
        None
        """
        try:
            self.cbar = self.create_colorbar(self.map)
        except Exception as e:
            # print(f"Failed to create colorbar using custom method: {e}")
            self.cbar = self.fig.colorbar(self.map, ax=self.ax, aspect=14)

        if self.cbar_name:
            self.cbar.set_label(self.cbar_name, fontsize=self.params.label_fontsize)

        if self.cbar_ticks:
            self.cbar.set_ticks(self.cbar_ticks)

        decimal_format = f"%.{self.cbar_decimal}f" if self.cbar_decimal > 0 else "%d"
        self.cbar.formatter = ticker.FormatStrFormatter(decimal_format)
        self.cbar.update_ticks()

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

    def batch_show_data(self):
        """
        批量显示数据，每个子图显示一个数据集。
        """
        for i in range(self.num_subplots):
            print(f"Processing subplot {i+1}")
            try:
                self.get_subfig(index=i+1)
                self.data = self.data_list[i]
                self.show_data()
            except Exception as e:
                print(f"Error processing subplot {i+1}: {e}")
                break

    def batch_add_labels_and_titles(self):
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

    def batch_add_colorbar(self):
        """
        批量添设置colorbar。
        """
        self.map = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        for i in range(self.num_subplots):
            self.get_subfig(index=i+1)
            self.add_colorbar()

    def batch_add_ticks_and_labels_simplified(self):
        """
        为所有子图批量设置标题、刻度和标签。
        在列的开始处添加y轴标签，在最后一行的子图添加x轴标签。
        """
        # 默认子标题，如果没有提供
        if self.subtitle is None:
            self.subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', 
                             '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
        
        for i in range(self.num_subplots):
            self.get_subfig(index=i + 1)
            self.ax.set_title(self.subtitle[i], fontsize=self.params.title_fontsize)

            self.ax.set_yticks(self.y_ticks)
            self.ax.set_xticks(self.x_ticks)

            # 仅在每行的第一列添加y轴标签和刻度标签
            if i % self.columns == 0:
                self.ax.set_ylabel(self.y_label, fontsize=self.params.label_fontsize)
                self.ax.set_yticklabels(self.y_tick_labels, fontsize=self.params.label_tick_size)
            else:
                self.ax.tick_params(axis='y', labelleft=False)

            # 在最后一行的所有子图中添加x轴标签和刻度标签
            if i >= self.num_subplots - self.columns:
                self.ax.set_xlabel(self.x_label, fontsize=self.params.label_fontsize)
                self.ax.set_xticklabels(self.x_tick_labels, fontsize=self.params.label_tick_size)
            else:
                self.ax.tick_params(axis='x', labelbottom=False)

            # 设置数值格式化
            self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"%{'.%df' % self.x_decimal}" if self.x_decimal > 0 else "%d"))
            self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%{'.%df' % self.y_decimal}" if self.y_decimal > 0 else "%d"))

    def add_unified_colorbar(self):
        """
        为所有子图添加统一的颜色条。设置颜色条的标签、刻度及其格式。

        颜色条的方向、长宽比及其它参数需事先在类属性中设置。

        返回:
        None, 直接修改了 self.cbar 属性。
        """
        try:
            # 添加颜色条到图形
            self.cbar = self.fig.colorbar(self.map, ax=self.axs, aspect=self.cbar_aspect, orientation=self.orientation)
        except Exception as e:
            print(f"Error adding colorbar: {e}")
            return

        # 设置颜色条的标签
        if self.cbar_name:
            self.cbar.set_label(self.cbar_name, fontsize=self.params.label_fontsize)

        # 设置颜色条的刻度
        if self.cbar_ticks:
            self.cbar.set_ticks(self.cbar_ticks)

        # 设置颜色条的数字格式
        decimal_format = f"%.{self.cbar_decimal}f" if self.cbar_decimal > 0 else "%d"
        self.cbar.formatter = ticker.FormatStrFormatter(decimal_format)
        self.cbar.update_ticks()

if __name__ == "__main__":

    from _gendata import gendata
    data_list = np.load('./data/gra_1_inv.npy')   # 生成测试数据
    print(data_list.shape)
    data_list = data_list.transpose((1, 2, 0))
    data_list = data_list[10:22, :, :]
    

    layout = 'double_column'  # 设置图形布局为双栏
    rows = 4                  # 设置子图行数为2
    columns = 3               # 设置子图列数为3
    projection3d = False      # 设置是否为3D图形，这里为2D

    data = data_list[0]  # 转置第一个数据集，通常用于调整数据的行列关系
    weight_height_ratio = data.shape[0]/data.shape[1]    # 计算图像的宽高比，保证图像的显示比例合理

    sf = StandardFigure()   # 创建一个标准图形实例

    # 设置基本参数，初始化图形布局和尺寸
    sf.set_base_params(layout=layout, rows=rows, columns=columns, 
                       weight_height_ratio=weight_height_ratio, 
                       projection3d=projection3d)

    cmap_name = 'seismic'          # 设置颜色映射为地震颜色
    equal_length_norm=True    # 设置是否使用等长度的归一化
    interpolation='none'      # 设置插值方式为无，保持数据像素的原始显示
    invert_yaxis = False
    colorbar = False
    sf.set_imshow_params(data_list=data_list, data=data, cmap_name=cmap_name, 
                         equal_length_norm=equal_length_norm,
                         interpolation=interpolation,
                         colorbar=colorbar)
    
    cbar_name = 'colorbar'
    cbar_ticks = None
    cbar_decimal = 1
    cbar_aspect = 30
    orientation = 'vertical'
    sf.set_colorbar_ticks_params(cbar_name=cbar_name, cbar_ticks=cbar_ticks, 
                                 cbar_decimal=cbar_decimal, cbar_aspect=cbar_aspect,
                                 orientation=orientation )
    
    x_label = 'x(km)'        # 设置x轴标签
    y_label = '深度/100'      # 设置y轴标签
    subtitle = None          # 未设置子标题（使用默认子标题）
    sf.set_label_and_title_params(x_label=x_label, y_label=y_label, subtitle=subtitle)

    x_tick_label_step, y_tick_label_step = 1.0, 1.0  # 设置刻度标签步长
    x_decimal = 1                                   # 设置x轴刻度的小数位数
    y_decimal = 0                                   # 设置y轴刻度的小数位数
    x_tick_interval, y_tick_interval = 10, 10       # 设置刻度间隔
    # 设置刻度参数
    sf.set_ticks_params(x_tick_label_step=x_tick_label_step, y_tick_label_step=y_tick_label_step,
                        x_decimal=x_decimal, y_decimal=y_decimal,
                        x_tick_interval=x_tick_interval, y_tick_interval=y_tick_interval)
    
    sf.create_figure()            # 创建图形
    # index = 1                   # 设置索引，可能用于选择特定的子图
    # sf.get_subfig(index=index)  # 获取指定索引的子图

    # 设置归一化参数
    sf.create_two_slope_norm(data=sf.data_list, equal_lengths=equal_length_norm)
    # sf.create_norm(vmin=-100, vmax=100)  

    # 生成数据图像
    sf.batch_show_data()  # 批量显示数据
    # sf.show_data()

    # 添加子图的标签和标题
    # sf.add_labels_and_title()
    # sf.batch_add_labels_and_titles() 

    # 批量添加刻度
    # sf.add_axis_ticks()
    # sf.batch_add_axis_ticks()  

    sf.batch_add_ticks_and_labels_simplified()

    # 添加colorbar
    # sf.batch_add_colorbar()
    sf.add_unified_colorbar()

    plt.savefig('test.tiff', dpi=300)     # 保存图像到文件

    plt.show()                   # 显示图像