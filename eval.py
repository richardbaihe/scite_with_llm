from sklearn.metrics import precision_recall_fscore_support
import os, json, argparse, glob

label_set = {"background", "result", "method"}
def compute_scores(path):
    print(path)
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
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=["background", "result", "method"], average=None, zero_division=0)
        labels = ["background", "result", "method"]
        print(",P,R,F1,S")
        for item in zip(labels, p, r, f, s):
            item = [str(x) for x in item]
            print(",".join(item))
        micro_rst = precision_recall_fscore_support(y_true, y_pred, labels=["background", "result", "method"], average='micro', zero_division=0)
        micro_rst = [str(x) if x else '0.0' for x in micro_rst]
        print('micro,' + ','.join(micro_rst))
        macro_rst = precision_recall_fscore_support(y_true, y_pred, labels=["background", "result", "method"], average='macro', zero_division=0)
        macro_rst = [str(x) if x else '0.0' for x in macro_rst]
        print('macro,' + ','.join(macro_rst))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--result_path', required=True, help='predict result path')
    # args = parser.parse_args()
    # compute_scores(args.result_path)

    files = glob.glob("./results/scicite/*")
    files = sorted(files)
    for file in files:
        compute_scores(file)
        print()
    print('Done!')