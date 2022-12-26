import os
import numpy as np
import pandas as pd
import collections
import json
import csv
from sample import *

datadir='.'
end = 'max_pool2d_nsight.csv'
files=[x for x in os.listdir(datadir) if x.endswith(end)]
files.sort()
files=[os.path.join(datadir,file) for file in files]
dfs={}


repeat_number = pd.read_csv(open("max_pool2d_repeat_number.csv"))
ALL_NUMBER = np.sum(repeat_number.values)
NUM_Metrics = 21
NUM_DEDU = len(repeat_number.values)


with open("max_pool2d_dedu.json", 'r') as f:
    shape_dict = json.load(f)

for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    dfs[tag]=pd.DataFrame()
    with open(file,'r') as f:
        df = pd.read_csv(file)
        df_group=pd.DataFrame()
        dft=pd.DataFrame(df, columns=['ID', 'Kernel Name','Metric Name', 'Metric Value'])
        
        dft['Metric Value'] = pd.to_numeric(dft['Metric Value'].str.replace(r',',''))
        dfmetric=pd.pivot_table(dft, index=['ID'], columns=['Metric Name'], values=['Metric Value'])
        dfmetric.to_csv("tmp.csv")
        kernel_name = dft['Kernel Name']
        kernel_name_dedu = []
        for k in range(0,len(kernel_name),NUM_Metrics):
            kernel_name_dedu.append(kernel_name[k])
        df_kernel_name_dedu=pd.DataFrame(kernel_name_dedu,columns=['Kernel Name'])

    
        csv_reader = csv.reader(open("tmp.csv"))
        os.remove("tmp.csv")
        count = 0
        with open("nsight_result.csv", 'w') as f:
            csv_writer = csv.writer(f)
            for line in csv_reader:
                if count != 0 and count != 2:
                    csv_writer.writerow(line)
                count += 1
            f.close
        count -= 2
        
        dfmetric = pd.read_csv(open("nsight_result.csv"))


        dfmetric['Time']=dfmetric['sm__cycles_elapsed.avg'] \
                        / (dfmetric['sm__cycles_elapsed.avg.per_second'] )
        dfmetric['Kernel Name'] = df_kernel_name_dedu['Kernel Name']

        df_list = ['Time', 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_dmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_dadd_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_fadd_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hfma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hadd_pred_on.sum', 
         'sm__inst_executed_pipe_tensor.sum',
         'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
         'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
         'lts__throughput.avg.pct_of_peak_sustained_elapsed',
         'dram__bytes.sum',
         'lts__t_bytes.sum',
         'l1tex__t_bytes.sum']

        df_dict = {i: []
        for i in df_list
         }
        
        cur_line = 0
        kernel_keys = "at::native::"
        k_start = False
        for index, kernel in dfmetric.iterrows():
            if kernel_keys in kernel['Kernel Name'].lower():
                for j in range(len(df_list)):
                    curnum = 0.0
                    for i in range(cur_line, cur_line+1):
                        curnum += float(dfmetric[df_list[j]][i])
                    df_dict[df_list[j]].append(curnum)
            cur_line += 1
        assert len(df_dict['Time']) == NUM_DEDU
        
        header = df_dict.keys()
        rows=pd.DataFrame(df_dict).to_dict('records')
        
        with open('deal.csv', 'w') as f:
            f.write(','.join(header))
            f.write('\n')
            for data in rows:
                f.write(",".join(str(data[h]) for h in header))
                f.write('\n')

        dfmetric = pd.read_csv(open("deal.csv"))
        os.remove("deal.csv")
        for i in df_list:
            dfmetric[i] = pd.to_numeric(dfmetric[i])


        dfmetric['CC FLOPs']= 2 * dfmetric['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
                            + 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
                            + 2 * dfmetric['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_hadd_pred_on.sum'] 

        dfmetric['TC FLOPs']= 512 * dfmetric['sm__inst_executed_pipe_tensor.sum']
        dfmetric['all FLOPs']= dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']
        
        dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(dfmetric['dram__bytes.sum'])
        dfmetric['AI L2'] = dfmetric['all FLOPs'].div(dfmetric['lts__t_bytes.sum'])
        dfmetric['AI L1'] = dfmetric['all FLOPs'].div(dfmetric['l1tex__t_bytes.sum'])

        dfmetric['GFLOP/s'] = dfmetric['all FLOPs']/ dfmetric['Time'] /1024/1024/1024
        dfmetric['TC GFLOP/s'] = dfmetric['TC FLOPs']/ dfmetric['Time'] /1024/1024/1024

        print(dfmetric['AI HBM'].values)
        print(np.sum(dfmetric['AI HBM'].values) / len(dfmetric['AI HBM'].values))

        dfmetric.to_csv('pd.csv')
        dfs[tag]=dfmetric

        in_infos, out_infos, kernel_infos = get_dedu_input_info()
        sizes, out_sizes, kernel_sizes  = get_dedu_input_data()

        # metric = 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'
        # metric = 'lts__throughput.avg.pct_of_peak_sustained_elapsed'
        # metric = 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed'
        metric = 'Time'
        # DEDU
        in_size_utilzation_dict = collections.OrderedDict()
        for idx, in_size in enumerate(sizes):
            # if idx :
            # if in_infos[idx][-7:] == "256x256" :
            # if in_infos[idx][-5:] == "42x42" :
            if in_size >= 2 * 64 * 400 * 592 and in_size <= 4 * 64 * 448 * 448:
                in_size += out_sizes[idx]*1.0/1e6 
                # if in_size in in_size_utilzation_dict.keys() and abs(in_size_utilzation_dict[in_size][0]-dfmetric[metric].values[idx]) > 10 and in_size_utilzation_dict[in_size][1] == in_infos[idx]:
                #     print("before", in_size_utilzation_dict[in_size], "and now",  [dfmetric[metric].values[idx], in_infos[idx], out_infos[idx]])
                in_size_utilzation_dict[in_size] = [dfmetric[metric].values[idx], in_infos[idx], out_infos[idx]]
        print('-----------------------------')

        in_size_utilzation_dict = collections.OrderedDict(sorted(in_size_utilzation_dict.items()))
        print(in_size_utilzation_dict)
        utilization_range_dict = {}
        # init
        column = 20
        start = 0
        step = 100 // column # 100%
        label_data = []
        for i in range(column):
            end = start + step
            key = str(start) + '-' + str(end)
            utilization_range_dict[key] = 0.0
            start += step
            label_data.append(key)

        for idx, util in enumerate(dfmetric[metric].values):
            key = int(util // step)
            utilization_range_dict[label_data[key]] += int(repeat_number.values[idx]) / ALL_NUMBER * 100
        print(utilization_range_dict)
        print(len(in_size_utilzation_dict))
        # -------------------------
        import matplotlib.pyplot as plt

        fontsize_tune = 8
        title_size = 10
        colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']

        allrown = 1
        fig, axes = plt.subplots(nrows=allrown, ncols=1, figsize=(20,13*allrown))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.8)    #subplots创建多个子图


        ########################## the first plt ################################
        rown = 0
        ax0 = axes

        axis = ax0
        # axis.set_ylabel('Memory Utilization(%)', fontsize=10)
        # axis.set_ylabel('L2 Cache Utilization(%)', fontsize=10)
        # axis.set_ylabel('L1 Cache Utilization(%)', fontsize=10)
        axis.set_ylabel('Consume Time(ms)', fontsize=10)
        data = list( i[0] for i in in_size_utilzation_dict.values())
        # names = list( (i[1]) for i in in_size_utilzation_dict.values())
        # names = list( (i[2]) for i in in_size_utilzation_dict.values())
        names = list( (i[1] + " " + i[2]) for i in in_size_utilzation_dict.values())
        axis.bar(names, data, color=colors)
        # axis.set_title('Memory Utilization of max_pool2d Function with Different Parameters'
                    # , fontsize=title_size)
        # axis.set_title('L2 Cache Utilization of max_pool2d Function with Different Parameters'
                    # , fontsize=title_size)
        # axis.set_title('L1 Cache Utilization of max_pool2d Function with Different Parameters'
                    # , fontsize=title_size)
        axis.set_title('Time duration of max_pool2d Function with Different Parameters'
                    , fontsize=title_size)
        axis.tick_params(axis='x', labelsize=fontsize_tune)
        axis.tick_params(axis='y', labelsize=fontsize_tune)
        axis.set_xticks(range(0,len(names),1))
        # axis.set_yticks(range(0,101,20))
        axis.set_xticklabels(names,rotation=30)
        plt.savefig('max_pool2d_memory_utilization.png',dpi=800)
        
        ########################## the 2nd plt ################################
        fig, axes = plt.subplots(nrows=allrown, ncols=1, figsize=(16,11*allrown))
        
        ax0 = axes
        axis = ax0
        axis.set_ylabel('Ratio(%)', fontsize=10) 
        names = list(utilization_range_dict.keys())
        data = list(utilization_range_dict.values())

        for x,y in zip(np.arange(len(names)),data):
            plt.text(x+0.05,y+0.05,'%.4f' %y, ha='center',va='bottom', fontdict={"size":4})

        axis.bar(names, data, color=colors)
        axis.set_title('Memory Utilization of max_pool2d Function with Different Parameters'
                    , fontsize=title_size)
        axis.tick_params(axis='x', labelsize=fontsize_tune)
        axis.tick_params(axis='y', labelsize=fontsize_tune)
        axis.set_xticks(range(0,len(names),1))
        axis.set_xticklabels(names,rotation=70)
        plt.savefig('max_pool2d_utilization_distribution.png',dpi=800)