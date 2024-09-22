# -*- coding: utf-8 -*-
"""
DanTagGen 模块用于基于输入提示生成标签。
"""

# 标准库导入
import os
import re
import traceback

# 第三方库导入
import torch
from transformers import  LlamaForCausalLM, LlamaTokenizer

# 本地模块导入
from kgen.formatter import seperate_tags, apply_format, apply_dtg_prompt
from kgen.metainfo import TARGET
from kgen.generate import tag_gen
from kgen.logging import logger
import kgen.models as models

# 正则表达式模式定义
re_attention = re.compile(
    r"""
    \\\(|
    \\\)|      # 转义的括号
    \\\[|
    \\]|       # 转义的方括号
    \\\\|      # 转义的反斜杠
    \\|        # 反斜杠
    \(|        # 左括号
    \[|        # 左方括号
    :\s*([+-]?[.\d]+)\s*\)|  # 匹配权重，例如 :1.5)
    \)|        # 右括号
    ]|         # 右方括号
    [^\\()\[\]:]+|  # 其他字符
    :
    """,
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    解析带有注意力标记的提示文本，并返回一个包含文本和对应权重的列表。

    参数：
        text (str): 包含注意力标记的提示文本。

    返回：
        List[List[str, float]]: 文本和对应权重的列表。
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        token = m.group(0)
        weight = m.group(1)

        if token.startswith('\\'):
            res.append([token[1:], 1.0])
        elif token == '(':
            round_brackets.append(len(res))
        elif token == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, token)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    # 合并具有相同权重的连续文本
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


class DanTagGen:
    """
    DanTagGen 类用于基于输入提示生成标签。
    """

    # 类级别常量
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_list = models.model_list

    def __init__(self):
        self.text_model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        加载模型和分词器。优先尝试使用 llama_cpp 加载本地模型，
        如果失败，则使用 transformers 从预训练模型加载。
        """
        ext_dir = os.path.dirname(os.path.realpath(__file__))
        models_dir = os.path.join(ext_dir, "models")
        all_model_files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        model_path = os.path.join(models_dir, all_model_files[-1]) if all_model_files else None

        if model_path:
            try:
                from llama_cpp import Llama
                self.text_model = Llama(
                    model_path,
                    n_ctx=384,
                    n_gpu_layers=100,
                    verbose=False,
                )
                self.tokenizer = LlamaTokenizer.from_pretrained("KBlueLeaf/DanTagGen-delta-rev2")
                return
            except ImportError:
                logger.warning("llama_cpp 未安装。尝试使用 transformers 加载模型。")
            except Exception as e:
                logger.error(f"使用 llama_cpp 加载模型时发生错误: {e}")
                logger.debug(traceback.format_exc())

        # 使用 transformers 加载模型
        try:
            self.text_model = LlamaForCausalLM.from_pretrained(
                "KBlueLeaf/DanTagGen-delta-rev2"
            ).eval().half().to(self.DEVICE)
            self.tokenizer = LlamaTokenizer.from_pretrained("KBlueLeaf/DanTagGen-delta-rev2")
        except Exception as e:
            logger.error(f"使用 transformers 加载模型失败: {e}")
            raise RuntimeError("模型加载失败。")

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数类型。
        """
        return {
            "required": {
                "model": (cls.model_list,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "format": ("STRING", {"default": """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,

<|general|>,

<|quality|>, <|meta|>, <|rating|>""", "multiline": True}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "temperature": ("FLOAT", {"default": 1.35, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
                "top_k": ("INT", {"default": 100}),
                "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "apply_DTG_formatting": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('prompt',)
    FUNCTION = 'execute'
    CATEGORY = 'utils'

    def execute(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
        top_p: float,
        top_k: int,
        apply_DTG_formatting: bool,
    ):
        """
        执行标签生成过程。

        参数：
            model (str): 模型名称。
            prompt (str): 用户输入的提示文本。
            width (int): 图像宽度。
            height (int): 图像高度。
            seed (int): 随机种子。
            tag_length (str): 标签长度。
            ban_tags (str): 禁用的标签列表。
            format (str): 标签格式模板。
            temperature (float): 生成温度。
            top_p (float): top-p 采样参数。
            top_k (int): top-k 采样参数。
            apply_DTG_formatting (bool): 是否应用 DTG 格式化。
        """
        # set_seed(seed)
        aspect_ratio = width / height
        prompt_without_extranet = prompt.strip()
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength != 1:
                for tag in part_tags:
                    strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        len_target = TARGET[tag_length]

        tag_map = seperate_tags(all_tags)
        dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)

        for llm_gen, extra_tokens,_ in tag_gen(
            self.text_model,
            self.tokenizer,
            dtg_prompt,
            tag_map["special"] + tag_map["general"],
            len_target,
            black_list,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=256,
            max_retry=5,
            seed=seed,
        ):
            pass

        tag_map["general"] += extra_tokens
        for cate in tag_map.keys():
            new_list = []
            for tag in tag_map[cate]:
                tag = tag.replace("(", r"\(").replace(")", r"\)")
                if tag in strength_map:
                    new_list.append(f"({tag}:{strength_map[tag]})")
                else:
                    new_list.append(tag)
            tag_map[cate] = new_list

        prompt_by_dtg = apply_format(tag_map, format)

        if not apply_DTG_formatting:
            user_prompt = prompt.strip()
            if user_prompt.endswith(','):
                user_prompt = user_prompt[:-1]

            for token in extra_tokens:
                token = token.strip()
                if token:
                    user_prompt += ", " + token
            prompt_by_dtg = user_prompt

        print(prompt_by_dtg)

        return (prompt_by_dtg,)


NODE_CLASS_MAPPINGS = {
    "DanTagGen": DanTagGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanTagGen": "DanTagGen",
}
