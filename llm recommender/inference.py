import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def my_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    stop_tokens = []
    sampling_params = SamplingParams(temperature=0.1, top_k=40, top_p=0.1, max_tokens=300,
                                     stop=stop_tokens)  # stop=stop_tokens
    print('sampleing =====', sampling_params)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)

    for kkk in ['MF', 'LightGCN', 'MixGCF', 'SGL']:  # , 'SASRec','BERT4Rec','CL4SRec''SGL','MF', 'LightGCN', 'SGL',
        data_path = f'/data/path/'
        INVALID_ANS = "[invalid]"
        res_ins = []
        res_answers = []
        problem_prompt = (
            "{instruction}"
        )
        with open(data_path, "r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                temp_instr = problem_prompt.format(instruction=item["inst"])
                res_ins.append(temp_instr)
        print('res_ins', res_ins)
        res_ins = res_ins[start:end]
        res_answers = res_answers[start:end]
        print('lenght ====', len(res_ins))
        batch_res_ins = batch_data(res_ins, batch_size=batch_size)
        result = []
        res_completions = []
        idx = 0
        for prompt in batch_res_ins:
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]
            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                local_idx = 'INDEX ' + str(idx) + ':'
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(generated_text)
                generated_text = generated_text.replace('\n', '').replace('    ', '')
                generated_text = local_idx + generated_text
                res_completions.append(generated_text)
                idx += 1
        print('res_completions', res_completions[:])
        def write_list_to_file(string_list, output_file):
            with open(output_file, 'w') as file:
                for item in string_list:
                    file.write(item + '\n')
        import pandas as pd
        df = pd.DataFrame(res_completions)
        df.to_csv(f'./res_completions{kkk}.txt', index=None, header=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default='/model/path/')  # model path
    parser.add_argument("--data_file", type=str,
                        default=f'/data/path/')  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=80)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    my_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size,
               tensor_parallel_size=args.tensor_parallel_size)
