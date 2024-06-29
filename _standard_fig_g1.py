import matplotlib.pyplot as plt 
from src._standardfig import StandardFigure
from src._gendata import gendata

def batch_process():
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
    cmap = 'seismic'          # 设置颜色映射为地震颜色
    equal_length_norm=True    # 设置是否使用等长度的归一化
    interpolation='none'      # 设置插值方式为无，保持数据像素的原始显示
    invert_yaxis = True       # 设置y轴翻转

    # 设置用于显示图像的参数
    fig1.set_imshow_params(data=data, cmap=cmap, 
                            equal_length_norm=equal_length_norm,
                            interpolation=interpolation,
                            invert_yaxis=invert_yaxis)

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

    plt.savefig('batch.tiff')     # 保存图像到文件

def single_process():

    data = gendata()   # 生成测试数据

    layout = 'single_column'  # 设置图形布局为双栏
    rows = 3                  # 设置子图行数为2
    columns = 1               # 设置子图列数为3
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

    def add_imshow(index, data):
        index = index # 设置索引，可能用于选择特定的子图
        fig1.get_subfig(index=index)  # 获取指定索引的子图，此行代码似乎未完成实现

        cmap = 'seismic'          # 设置颜色映射为地震颜色
        equal_length_norm=True    # 设置是否使用等长度的归一化
        interpolation='none'      # 设置插值方式为无，保持数据像素的原始显示
        invert_yaxis = True       # 设置y轴翻转

        # 设置用于显示图像的参数
        fig1.set_imshow_params(data=data, cmap=cmap, 
                                equal_length_norm=equal_length_norm,
                                interpolation=interpolation,
                                invert_yaxis=invert_yaxis)
        # fig1.set_batch_norm(vmin=-100, vcenter=0, vmax=100)  # 设置归一化参数
        fig1.show_data()  # 批量显示数据

        x_label = 'x(km)'        # 设置x轴标签
        y_label = '深度/100'     # 设置y轴标签
        subtitle = 'index' + str(index)         # 设置子标题（使用默认子标题）
        fig1.set_label_and_title_params(x_label=x_label, y_label=y_label, subtitle=subtitle)
        fig1.add_labels_and_title()  # 批量设置子图的标签和标题

        x_tick_label_step, y_tick_label_step = 1.0, 2.0  # 设置刻度标签步长
        max_bins_x, max_bins_y = 5, 4                   # 设置最大刻度数
        x_decimal = 1                                   # 设置x轴刻度的小数位数
        y_decimal = 0                                   # 设置y轴刻度的小数位数

        (y_num, x_num) = data.shape                  # 获取第二个数据集的形状
        x_tick_interval, y_tick_interval = 10, 10       # 设置刻度间隔  

        # 设置刻度参数
        fig1.set_ticks_params(x_tick_label_step=x_tick_label_step, y_tick_label_step=y_tick_label_step,
                            x_decimal=x_decimal, y_decimal=y_decimal,
                            x_num=x_num, y_num=y_num, x_tick_interval=x_tick_interval, y_tick_interval=y_tick_interval)
        fig1.add_axis_ticks()  # 批量添加刻度

    add_imshow(index = 1, data = data[0])
    add_imshow(index = 3, data = data[1])

    fig1.fig.tight_layout()      # 自动调整布局，避免内容重叠

    plt.savefig('singel.tiff')     # 保存图像到文件


if __name__ == "__main__":

    batch_process()
    single_process()

    plt.show()
