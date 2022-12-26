
import matplotlib.pyplot as plt


def func_pie():
    def my_autopct(pct):
        return ('%.1f' % pct + '%') if pct > 2.5 else ''

    labels = ['ReLU', 'GeLU', 'LeakyRelu', 'Hardtanh',
              'Sigmoid', 'Softmax', 'LogSoftmax', 'GLU']
    appear_in_network = [1065, 40, 39, 24, 188, 225, 36, 1]
    call_times = [646227, 3987, 19599, 4310, 14346, 8373, 313, 5]
    explode = (0.01, 0, 0, 0, 0, 0 ,0.005, 0)  # 元素凸出距离
    colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y', 'lightgreen', 'orange', 'slateblue']
    # 饼图绘制函数
    plt.figure(figsize=(11.4, 6))
    plt.pie(call_times, explode=explode, radius=0.3, labels=labels, colors=colors, \
            autopct=my_autopct, shadow=False, pctdistance=0.8, \
            startangle=90, textprops={'fontsize': 11, 'color': 'w'})
    # plt.title('abc')
    plt.axis('equal')
    plt.legend(loc=1)
    plt.legend()
    plt.savefig('pie.png', dpi=800)

func_pie()
