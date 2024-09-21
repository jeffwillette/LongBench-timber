import os
from datasets import load_dataset
import torch
import deepspeed
import json
from cascade.models.cascading_cache import CascadingKVCache
from transformers import AutoTokenizer, AutoConfig
from cascade.models.cascade_attention import sample_monkeypatch
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.multiprocessing as mp
import transformers
from typing import Union

from cascade.models.llama.modeling_llama import LlamaForCausalLM
from cascade.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

# from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default=None,
                        choices=[
                            "llama3.1-8b-instruct",
                            "llama2-7b-chat-4k",
                            "longchat-v1.5-7b-32k",
                            "xgen-7b-8k",
                            "internlm-7b-8k",
                            "chatglm2-6b",
                            "chatglm2-6b-32k",
                            "chatglm3-6b-32k",
                            "vicuna-v1.5-7b-16k",
                            "llama2-7b-chat-32k",
                            "llama2-13b-chat-32k",
                            "qwen2-14b-chat-32k",
                            "qwen2-7b-chat-32k",
                            "qwen2-7b-instruct",
                            "llama3-8b-8k",
                            "llama3-8b-16k",
                            "llama3-8b-262k",
                            "phi3-3b-128k",
                            "llama1.3b",
                            "qwen0.5b",
                        ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--method',
                        type=str,
                        default=os.getenv("ATTENTION_METHOD", "none"),
                        choices=['none', 'vanilla', 'hip', 'streaming_llm', 'bigbird'])
    parser.add_argument('--sinks', type=int, default=None)
    parser.add_argument('--cascades', type=int, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--comment', type=str, default="none")
    parser.add_argument('--local_rank', default=0, type=int)

    parser.add_argument('--stride', type=int, default=None)

    args = parser.parse_args(args)

    if args.method in ['streaming_llm', 'hip']:
        assert args.cascades is not None
        assert args.sinks is not None
        assert args.window is not None

    assert args.method == ATTENTION_METHOD
    args.world_size = torch.cuda.device_count()

    return args


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]\n{prompt}\n[/INST]\n\n"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "qwen2" in model_name:
        prompt = f'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
    elif "llama3" in model_name:
        prompt = f'<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif "phi3" in model_name:
        raise Exception('phi3 not supported yet on vllm-hip')
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


ATTENTION_METHOD = os.getenv('ATTENTION_METHOD', 'none')


class StoppingCriteriaSub(transformers.StoppingCriteria):

    def __init__(self, stops=[], tokenizer=None, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(
                    last_token):
                return True
        return False


def get_pred(
    rank,
    world_size,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    model2path,
    out_path,
    args,
):

    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device,
        max_length,
        args,
    )
    mdl = model.module.model

    for json_obj in tqdm(data, desc=dataset):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt,
                                     truncation=False,
                                     return_tensors="pt").input_ids[0]

        if args.method == ("vanilla" and "truncate" in args.comment):
            max_length = int(2 ** int(np.log2(tokenized_prompt.size(0) / 4) // 1)) - max_gen
            # print(f"{tokenized_prompt.size(0)=} {max_length=}")

        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            # if we are truncating to be equivalent with the cascade models, reset the halfway mark
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            # print(f"len after slicing: {input['input_ids'].size()=}")
        context_length = input.input_ids.shape[-1]

        if ATTENTION_METHOD == 'streaming_llm':
            if dataset == "samsum":  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                raise Exception()
                with torch.inference_mode():
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=context_length + 1,
                        eos_token_id=[
                            tokenizer.eos_token_id,
                            tokenizer.encode("\n",
                                             add_special_tokens=False)[-1]
                        ],
                    )[0]
            else:
                if 'llama' in model_name:
                    stop_words = ["<|eot_id|>"]
                elif 'qwen' in model_name:
                    stop_words = ["<|im_end|>"]

                stop_words_ids = [
                    tokenizer(
                        stop_word,
                        return_tensors='pt',
                        add_special_tokens=False)['input_ids'].squeeze()
                    for stop_word in stop_words
                ]
                stopping_criteria = transformers.StoppingCriteriaList([
                    StoppingCriteriaSub(
                        stops=stop_words_ids,
                        tokenizer=tokenizer,
                        device="cuda",
                    )
                ])

                input_ids = input["input_ids"]
                # input_ids = input_ids[:, :50]  # for debugging
                # context_length = 50
                # mask = input["attention_mask"]

                max_seq_len = int(2 ** int(np.log2(input_ids.size(1) / 4) // 1))
                max_seq_len = min(max_seq_len, max_length)
                print(f"{max_seq_len=}")
                window = max_seq_len // args.cascades

                # window = mdl.config._window // mdl.config._cascades
                # max_seq_len = mdl.config._window

                past_key_values = CascadingKVCache(
                    window,
                    num_sink_tokens=mdl.config._sinks,
                    max_batch_size=mdl.config._batch_size,
                    heads=mdl.config.num_key_value_heads // args.world_size,
                    dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
                    max_seq_len=max_seq_len,
                    dtype=torch.float16,
                    device=mdl.embed_tokens.weight.device,
                    cascade_func=mdl.config._cascade_func,
                    head_reduction=mdl.config._head_reduction,
                    layers=len(mdl.layers),
                )

                output = model.generate(
                    input_ids=input_ids,
                    # attention_mask=mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    past_key_values=past_key_values,
                )[0]

            pred = tokenizer.decode(output[context_length:],
                                    skip_special_tokens=True)

            past_key_values.reset()

        else:
            if 'llama' in model_name:
                stop_words = ["<|eot_id|>"]
            elif 'qwen' in model_name:
                stop_words = ["<|im_end|>"]

            if args.method == "bigbird":
                print(f"input ids before setting max seq length in bigbird generate: {input['input_ids'].size(1)=}")
                max_seq_len = int(2 ** int(np.log2(input['input_ids'].size(1) / 4) // 1)) - max_gen
                max_seq_len = min(max_seq_len, 32768)
                print(f"{max_seq_len=}")
                for lyr in mdl.layers:
                    if args.method == "bigbird":
                        lyr.self_attn.config._bb_window = max_seq_len
                    elif args.method == "snapkv":
                        lyr.self_attn.config.max_capacity_prompt = max_seq_len

            stop_words_ids = [
                tokenizer(
                    stop_word,
                    return_tensors='pt',
                    add_special_tokens=False)['input_ids'].squeeze()
                for stop_word in stop_words
            ]
            stopping_criteria = transformers.StoppingCriteriaList([
                StoppingCriteriaSub(
                    stops=stop_words_ids,
                    tokenizer=tokenizer,
                    device=device,
                )
            ])

            input_ids = input["input_ids"]

            output = model.generate(
                input_ids=input_ids,
                # attention_mask=mask,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                past_key_values=None,
            )[0]

            pred = tokenizer.decode(output[context_length:],
                                    skip_special_tokens=True)

        pred = post_process(pred, model_name)

        if args.local_rank == 0:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": pred,
                        "answers": json_obj["answers"],
                        "all_classes": json_obj["all_classes"],
                        "length": json_obj["length"]
                    },
                    f,
                    ensure_ascii=False)
                f.write('\n')
                f.flush()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, seq_len, args):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    ModelClass = LlamaForCausalLM
    if 'qwen' in model_name:
        ModelClass = Qwen2ForCausalLM

    if ATTENTION_METHOD == 'streaming_llm':
        config = AutoConfig.from_pretrained(path)
        config.attn_implementation = config._attn_implementation = 'eager'
        config._batch_size = 1
        config._sinks = args.sinks
        config._cascades = args.cascades
        config._window = args.window
        config._cascade_func = "pow2"
        config._head_reduction = "mean"
        config._cascade_stride = 512
        config.max_position_embeddings
        # print(f"{config.max_position_embeddings=}")
        config._method = "sink"
        config.world_size = args.world_size
        config._cascade_func = "pow2"
        config._head_reduction = "max"
        config._homogeneous_heads = False
        config._do_og_pos = False

        model = ModelClass.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model = sample_monkeypatch(model)

    else:
        config = AutoConfig.from_pretrained(path)
        config.attn_implementation = config._attn_implementation = 'flash_attention_2'
        if args.method == "bigbird":
            config._method = "bigbird"
        else:
            config._method = "vanilla"

        model = ModelClass.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

    model = deepspeed.init_inference(
        model,
        tensor_parallel={"tp_size": args.world_size},
        replace_with_kernel_inject=False,
        dtype=torch.float16,
        max_out_tokens=seq_len,
    )

    model = model.cuda()
    model.eval()

    return model, tokenizer


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    # this is not the real world size since we are doing tensor parallel and do not want to
    # parallelize over the dataset. Will set real world size in model loader
    world_size = 1
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    print(model2maxlen)
    model_name = args.model

    # define your model
    max_length = model2maxlen[model_name] if args.stride is None else args.stride
    if args.method == "streaming_llm":
        # for streaming llm, we will never run out of positional embeddings for the quarter-ctx experiment.
        # as the longest prompt in longbench is 65,370. Therefore, the max prompt should be somewhere around 8K
        # so set this max length for all streaming llm models which should not truncate the inputs.
        max_length = 131072

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = [
            'narrativeqa',
            'hotpotqa',
            'gov_report',
            'multi_news',
            'qasper',
            '2wikimqa',
        ]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # predict on each dataset
    os.makedirs("pred", exist_ok=True)
    os.makedirs("pred_e", exist_ok=True)

    for dataset in datasets:
        pred_root_name = None
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            pred_root_name = 'pred_e'
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            pred_root_name = 'pred'

        if args.method in ['vanilla', 'bigbird']:
            pred_root = f"{pred_root_name}/{model_name}_{args.method}_comment_{args.comment}"
        elif args.method in ['streaming_llm', 'hip']:
            pred_root = f"{pred_root_name}/{model_name}_{args.method}_window_{args.window}_cascades_{args.cascades}_sinks_{args.sinks}_comment_{args.comment}"
        else:
            raise Exception()

        os.makedirs(pred_root, exist_ok=True)
        out_paths = [
            os.path.join(pred_root, f'{dataset}-{i}.jsonl')
            for i in range(world_size)
        ]
        devices = [torch.device(f'cuda:{i}') for i in range(world_size)]

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        # for debugging
        # data_subsets = [v[:2] for v in data_subsets]

        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    world_size,
                    data_subsets[rank],
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    devices[rank],
                    model_name,
                    model2path,
                    out_paths[rank],
                    args,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
