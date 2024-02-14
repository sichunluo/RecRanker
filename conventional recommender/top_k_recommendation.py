from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    graph_baselines = ['LightGCN','DirectAU','MF']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec','DuoRec','BERT4Rec']
    model = 'SGL'
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ml-100k', help='')
    parser.add_argument("--model", type=str, default='SASRec', help='')
    parser.add_argument("--aug_type", type=str, default='0', help='')
    args = parser.parse_args()
    model = args.model
    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    conf.__setitem__('training.set',f'./dataset/{args.dataset}/train_set_seq.txt')
    conf.__setitem__('valid.set',f'./dataset/{args.dataset}/valid_set_seq.txt')
    conf.__setitem__('test.set',f'./dataset/{args.dataset}/test_set_seq.txt')
    conf.__setitem__('aug_type',args.aug_type)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
