from sklearn.metrics import precision_recall_fscore_support
import os, json, argparse, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors

label_set = {"background", "result", "method"}
type_map = {'p':0, 'r':1, "f1":2}
def compute_scores(path, result_keep, type):
    path_arr = path.replace('./results/scicite/engine_code-cushman-001_prompt_','').replace('_test.jsonl','').split('_shot_')
    print(path_arr)
    y_true = []
    y_pred = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            result = json.loads(line)
            label = result["label"]
            predict = result["generation"][result["input_length"]:].lower()
            if '"' in predict:
                predict = predict.split('"')[0]
            else:
                predict = predict[:len(label)]
            if predict not in label_set:
                predict = 'other'
            y_true.append(label)
            y_pred.append(predict)
        macro_rst = precision_recall_fscore_support(y_true, y_pred, labels=["background", "result", "method"], average='macro', zero_division=0)        
        macro_rst = macro_rst[type_map[type]:type_map[type]+1]
        macro_rst = [round(x, 2) if x else '0.0' for x in macro_rst]
        # print('macro,' + ','.join(macro_rst))
        if path_arr[0] in result_keep:
            result_keep[path_arr[0]][path_arr[1]] = macro_rst[0]

def plot_func(data, order, type, color_map, axis_map, temp_name_map):
    mcolors.TABLEAU_COLORS
    # set width of bar
    barWidth = 0.15
    fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br = None
    br1 = np.arange(3)
    # br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]

    # Make the plot
    for temp in order:
        scores =np.array([ data[temp][str(x)] for x in range(0,7,3)])
        print(temp, temp_name_map[temp], scores)
        plt.bar(br1, scores, color = color_map[temp], width = barWidth,
                edgecolor ='grey', label =temp_name_map[temp])
        br1 = [x + barWidth for x in br1]
    # Adding Xticks
    plt.xlabel('Shots', fontweight ='bold', fontsize = 20)
    plt.ylabel(axis_map[type], fontweight ='bold', fontsize = 20)
    plt.xticks([r + barWidth for r in range(len(data[order[0]]))],
            ['0','3','6'],fontsize = 20)
    plt.yticks(fontsize = 20)

    plt.legend(fontsize=16)
    plt.savefig(f'figures/{type}.pdf')

    # plt.show()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--result_path', required=True, help='predict result path')
    # args = parser.parse_args()
    # compute_scores(args.result_path)

    type = 'f1'
    result_keep = dict()
    temp_set = ['multi_choice_single_line', 'json', 'cloze_style_logprobs', 'json_logprobs']
    for temp in temp_set:
        result_keep[temp] = defaultdict(float)
    files = glob.glob("./results/scicite/*")
    files = sorted(files)
    for file in files:
        compute_scores(file, result_keep, type)
        print()

    color_map = {
        'multi_choice_single_line':'r', 
        'json':'g', 
        'cloze_style_logprobs':'y', 
        'json_logprobs':'b'
        }
    
    axis_map = {
        'f1':'F1', 
        'p':'Precision', 
        'r':'Recall', 
        }
    temp_name_map = {
        'multi_choice_single_line':'MCQ-free-decoding', 
        'json':'JSON-free-decoding', 
        'cloze_style_logprobs':'MCQ-force-decoding-calibrated', 
        'json_logprobs':'JSON-force-decoding'
        }

    print(result_keep)
    

    print('Done!')

    plot_func(result_keep, temp_set, type, color_map, axis_map, temp_name_map)
