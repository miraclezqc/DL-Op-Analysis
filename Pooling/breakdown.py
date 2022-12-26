
import matplotlib.pyplot as plt


def func_pie():
    def my_autopct(pct):
        return ('%.1f' % pct + '%') if pct > 2 else ''

    labels = ['adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
              'avg_pool2d', 'avg_pool3d', 'max_pool2d', 'max_pool3d']
    appear_in_network = [369, 12, 9, 2, 70, 4, 780, 22]
    call_times = [15097, 901, 245, 880, 2329, 31, 5506, 1533]
    explode = (0.01, 0, 0, 0, 0, 0 ,0.005,0)  # 元素凸出距离
    colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y', 'lightgreen', 'orange', 'slateblue']
    # 饼图绘制函数
    plt.figure(figsize=(11.4, 6))
    plt.pie(call_times, explode=explode, radius=0.3, labels=labels, colors=colors, \
            autopct=my_autopct, shadow=False, pctdistance=0.8, \
            startangle=90, textprops={'fontsize': 14, 'color': 'w'})
    # plt.title('abc')
    plt.axis('equal')
    plt.legend(loc=1)
    plt.legend()
    plt.savefig('pie.png', dpi=800)

func_pie()
