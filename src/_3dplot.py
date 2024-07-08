import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import copy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
try:
    from src._standardfig import StandardFigure
except:
    from _standardfig import StandardFigure

class Plot3D(StandardFigure):
    
    @staticmethod
    def cube_vertices(adjustment, x1, x2, y1, y2, z1, z2):
        """
        定义立方体的八个顶点，并生成立方体的面。

        参数:
        adjustment: 调整值，用于略微缩小立方体的尺寸以避免重叠。
        x1, x2: x轴上的两个坐标值。
        y1, y2: y轴上的两个坐标值。
        z1, z2: z轴上的两个坐标值。

        返回值:
        faces: 组成立方体的各个面。
        """

        # 调整立方体的各个坐标
        x1, x2, y1, y2, z1, z2 = x1 + adjustment, x2 - adjustment, y1 + adjustment, y2 - adjustment, z1 + adjustment, z2 - adjustment

        # 定义立方体的八个顶点
        vertices = np.array([
            [x1, y1, z1],
            [x1, y1, z2],
            [x1, y2, z1],
            [x1, y2, z2],
            [x2, y1, z1],
            [x2, y1, z2],
            [x2, y2, z1],
            [x2, y2, z2]
        ])

        # 定义组成立方体的面
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[1], vertices[3], vertices[2]],
            [vertices[7], vertices[6], vertices[4], vertices[5]],
            [vertices[7], vertices[3], vertices[1], vertices[5]],
            [vertices[0], vertices[2], vertices[6], vertices[4]]
        ]

        return faces

    @staticmethod
    def add_cube(ax, faces, face_color, data_threshold, edge_color, show_empty=False, transparent=True):
        """
        在3D图中添加一个立方体。

        参数:
        ax: 3D轴对象。
        faces: 立方体的面列表。
        face_color: 面的颜色。
        data_threshold: 数据阈值，决定是否显示立方体。
        edge_color: 边缘的颜色。
        show_empty: 是否显示空立方体，默认为 False。
        transparent: 是否透明，默认为 True。
        """
        face_color_copy = copy(face_color)
        if not transparent:
            face_color[3] = 1

        # 添加满足阈值的立方体
        for face in faces:
            if face_color_copy[3] > data_threshold:
                poly = Poly3DCollection([face])
                poly.set_facecolor(face_color)
                poly.set_edgecolor(edge_color)
                ax.add_collection3d(poly)

        # 如果显示空立方体，则添加低透明度的立方体
        if show_empty:
            for face in faces:
                if face_color_copy[3] < data_threshold:
                    poly = Poly3DCollection([face])
                    # face_color[3] = 1e-2
                    # poly.set_facecolor(face_color)
                    poly.set_edgecolor((0.1, 0.1, 0.1, 1e-2))
                    ax.add_collection3d(poly)

    @staticmethod
    def set_face_color(norm, colormap, data_value, transparency_adjustment):
        """
        设置颜色，包括透明度调整。

        参数:
        norm: 归一化对象。
        colormap: 颜色映射。
        data_value: 数据值。
        transparency_adjustment: 透明度调整参数。

        返回值:
        face_color: 带有透明度的颜色。
        """
        face_color = colormap(norm(data_value))
        face_color = list(face_color)
        face_color[3] = (norm(data_value) - 0.5) * 2
        face_color[3] = face_color[3] ** transparency_adjustment

        return face_color

    @staticmethod
    def set_axis_limits(ax, x_limit, y_limit, z_limit):
        """
        设置3D图的坐标轴限制。

        参数:
        ax: 3D轴对象。
        x_limit: x轴的限制。
        y_limit: y轴的限制。
        z_limit: z轴的限制。
        """
        ax.set_xlim([0, x_limit + 1])
        ax.set_ylim([0, y_limit + 1])
        ax.set_zlim([z_limit + 1, 0])

    def set_pane_color(self):
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def add_colorbar_3d(self, shrink=0.6, pad=0.15):
        """
        为3D图添加颜色条。

        参数:
        ax: 3D轴对象。
        norm: 归一化对象。
        colormap: 颜色映射。
        shrink: 颜色条的缩放比例，默认为 0.6。
        pad: 颜色条与图像的间距，默认为 0.15。

        返回值:
        cbar: 颜色条对象。
        """
        bounds = [self.norm.vmin, self.norm.vcenter, self.norm.vmax]
        mappable = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        cbar = self.fig.colorbar(mappable, ax=self.ax, shrink=shrink, pad=pad, ticks=bounds)
        # cbar = add_colorbar(ax)
        return cbar


    def auto_adjust_colorbar(self, mappable, orientation='vertical', aspect_ratio=14, padding_fraction=0.5, **kwargs):
        """
        自动计算并添加颜色条，适用于2D和3D图形。

        参数:
        ax: 轴对象，可以是2D或3D。
        mappable: 可映射的对象，例如来自`imshow`、`scatter`或`plot_surface`的对象。
        orientation: 颜色条的方向，默认为'vertical'。
        aspect_ratio: 颜色条的纵横比，默认为20。
        padding_fraction: 颜色条与图形的间距比例，默认为0.5。
        **kwargs: 传递给colorbar的其他参数。

        返回值:
        colorbar: 添加到图像的颜色条对象。
        """
        # 创建一个与图像关联的可分隔轴对象
        ax_pos = self.ax.get_position()

        print(ax_pos, ax_pos.height)
        width = ax_pos.height/aspect_ratio
        print(width)
        cb_ax_pos = [ax_pos.x1+width*padding_fraction, 0.1*ax_pos.height+ax_pos.y0, width, 0.8*ax_pos.height]
        print(cb_ax_pos)
        cb_ax = self.fig.add_axes(cb_ax_pos)
        colorbar = self.fig.colorbar(mappable,ax=self.ax, cax=cb_ax, orientation=orientation, **kwargs)
        return colorbar
    
    def make_3d_figure(self, data, colormap_name, size_adjustment, transparency_adjustment, data_threshold, edge_color, show_empty=False, transparent=True):
        """
        创建3D图并添加数据立方体。

        参数:
        ax: 3D轴对象。
        data: 数据数组。
        colormap_name: 颜色映射名称。
        norm: 归一化对象。
        size_adjustment: 尺寸调整参数。
        transparency_adjustment: 透明度调整参数。
        data_threshold: 数据阈值。
        edge_color: 边缘颜色。
        show_empty: 是否显示空立方体，默认为 False。
        transparent: 是否透明，默认为 True。
        """
        
        self.cmap = self.get_scico_colormap(colormap_name)

        for (i, j, k), value in np.ndenumerate(data):
            face_color = self.set_face_color(self.norm, self.cmap, value, transparency_adjustment)
            x1, x2, y1, y2, z1, z2 = i, i + 1, j, j + 1, k, k + 1
            faces = self.cube_vertices(size_adjustment, x1, x2, y1, y2, z1, z2)
            edge_color[3] = face_color[3] / 10
            # print(face_color, edge_color)
            self.add_cube(self.ax, faces, face_color, data_threshold, edge_color, show_empty, transparent)
            self.set_axis_limits(self.ax, i + 1, j + 1, k + 1)
            self.ax.set_box_aspect([i + 1, j + 1, k + 1])

# def generate_gif(ax, save_path):
#     """
#     生成并保存3D图的GIF动画。

#     参数:
#     ax: 3D轴对象。
#     save_path: 动画保存路径。
#     """
    
#     from matplotlib.animation import FuncAnimation

#     def update(frame):
#         ax.view_init(elev=30, azim=frame)
#         return fig,

#     # 创建动画
#     animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 10), interval=50)

#     # 保存动画
#     animation.save(save_path + '.gif', writer='Pillow')

if __name__ == "__main__":

    layout = 'single_column'  # 设置图形布局为双栏
    rows = 1                  # 设置子图行数为2
    columns = 1               # 设置子图列数为3
    projection3d = True      # 设置是否为3D图形，这里为2D
    weight_height_ratio = 1   # 设置图片的宽高比，通常用于保证图像的显示比例合理

    fig1 = Plot3D()   # 创建一个标准图形实例

    # 设置基本参数，初始化图形布局和尺寸
    fig1.set_base_params(layout=layout, rows=rows, columns=columns, 
                         weight_height_ratio=weight_height_ratio, 
                         projection3d=projection3d)
    fig1.create_figure()      # 创建图形

    data = np.load('./data/gra_1_inv.npy')

    # data_max = np.max(data)

    # vmin = -data_max
    # vcenter = 0
    # vmax = data_max
    # norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap_name = 'seismic'          # 设置颜色映射为地震颜色
    fig1.equal_length_norm=True    # 设置是否使用等长度的归一化
    fig1.norm = fig1.create_two_slope_norm(data, equal_lengths=True)
    
    size_adj = 0
    data_threshold = 0.3
    transparent_adj = 1
    show_empty = False
    transparent = True

    edge_color = [0.1, 0.1, 0.1, 0.01]

    index = 1
    fig1.get_subfig(index=index)
    print(fig1.ax)
    fig1.make_3d_figure(data=data, colormap_name=cmap_name, 
                        size_adjustment=size_adj, transparency_adjustment=transparent_adj,
                        data_threshold = data_threshold, edge_color = edge_color,
                        show_empty=show_empty, transparent=transparent)

    fig1.set_pane_color()

    x_label = 'X/m'
    y_label = 'Y(km)'
    z_label = 'Z*1'
    subtitle = 'Fig_test'
    fig1.set_label_and_title_params(x_label=x_label, y_label=y_label, z_label=z_label, subtitle=subtitle)
    fig1.add_labels_and_title()  # 批量设置子图的标签和标题
    # ax.zaxis.set_rotate_label(False)  # 禁用 z 轴标签的旋转




    # fig1.ax.invert_zaxis()
    # # cbar = add_colorbar(ax, norm, cmap, shrink = 0.6, pad = 0.15)
    elevation = 20
    azimuth = 150
    bounds = [fig1.norm.vmin, fig1.norm.vcenter, fig1.norm.vmax]
    mappable = plt.cm.ScalarMappable(norm=fig1.norm, cmap=fig1.cmap)
    # cbar = fig1.create_colorbar(fig1.ax)
    # cbar = fig1.auto_adjust_colorbar(mappable=mappable)

    cbar = fig1.add_colorbar_3d(shrink=0.5, pad=0.2)
    fig1.ax.zaxis._axinfo['juggled'] = (1,2,0)
    # ax.view_init(elev=elevation, azim=azimuth)
    # fig1.fig.tight_layout() 
    ax_pos = fig1.ax.get_position()

    fig1.ax.set_position([ax_pos.x0+0.2*ax_pos.width, ax_pos.y0, ax_pos.width*0.9, ax_pos.height])  # 这些值也是以图形窗口的比例为单位

    savepath = 'rotation'
    # generate_gif(ax, savepath)

    # def onclick(event):
    #     print(f'你点击的位置是：({event.xdata}, {event.ydata})')

    # fig.canvas.mpl_connect('button_press_event', onclick)

    plt.savefig('./figure/3dplottest.tiff')
    plt.show()