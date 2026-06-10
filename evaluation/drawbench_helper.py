# drawbench_helper.py
import os
import pandas as pd
import urllib.request
import random
import json

class DrawBenchPromptGenerator:
    CSV_URL = "https://huggingface.co/datasets/sayakpaul/drawbench/resolve/main/DrawBench%20Prompts%20-%20Sheet1.csv"
    
    def __init__(self, csv_path="DrawBench_Prompts.csv", json_path="DrawBench_Prompts.json"):
        self.csv_path = csv_path
        self.json_path = json_path
        self.tasks = {}      # {category: list of (idx, prompt)}
        self.idx_to_prompt = {}  # {idx: prompt}
        self._load_or_download()
    
    def _download_csv(self):
        print("正在下载 DrawBench prompt CSV ...")
        urllib.request.urlretrieve(self.CSV_URL, self.csv_path)
        print(f"下载完成，保存为 {self.csv_path}")
    
    def _load_or_download(self):
        if os.path.exists(self.json_path):
            print(f"加载已有 JSON 文件 {self.json_path}")
            with open(self.json_path, "r") as f:
                data = json.load(f)
            
            self.tasks = {}
            self.idx_to_prompt = {}
            current_idx = 0
            for task, prompts in data.items():
                if isinstance(prompts[0], list) or isinstance(prompts[0], tuple):
                    # 新格式 (idx, prompt)
                    self.tasks[task] = [(int(p[0]), p[1]) for p in prompts]
                else:
                    # 旧格式，只存 prompt 字符串
                    self.tasks[task] = [(current_idx + i, p) for i, p in enumerate(prompts)]
                for idx, p in self.tasks[task]:
                    self.idx_to_prompt[idx] = p
                current_idx += len(prompts)
            return
        
        if not os.path.exists(self.csv_path):
            self._download_csv()
        
        df = pd.read_csv(self.csv_path)
        if "Prompts" not in df.columns or "Category" not in df.columns:
            raise ValueError("CSV 文件缺少 'Prompts' 或 'Category' 列")
        
        self.tasks = {}
        self.idx_to_prompt = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            task = row["Category"]
            prompt = row["Prompts"]
            if task not in self.tasks:
                self.tasks[task] = []
            self.tasks[task].append((idx, prompt))
            self.idx_to_prompt[idx] = prompt
        
        # 保存 JSON
        with open(self.json_path, "w") as f:
            json.dump(self.tasks, f, indent=2)
        print(f"保存 JSON 文件: {self.json_path}")
    
    def sample_prompts(self, x=None, seed=0):
        """
        按类别尽量均匀采样 x 条 prompt，返回 (idx, prompt)
        支持 seed 保证可重复
        
        Args:
            x (int or None): 需要抽取的总 prompt 数，如果 None 则返回全部 prompt
            seed (int): 随机种子
        
        Returns:
            list[(idx, prompt)]
        """
        all_pairs = self.get_all_prompts()
        if x is None:
            print(f"返回全部 {len(all_pairs)} 条 prompt")
            return all_pairs
        if x >= len(all_pairs):
            print(f"请求数量 {x} 超过总数 {len(all_pairs)}，返回全部 prompt")
            return all_pairs
        rng = random.Random(seed)
        num_tasks = len(self.tasks)
        base = x // num_tasks
        remainder = x % num_tasks
        sampled = []
        sampled_counter = {}
        leftover = []
        
        for i, (task, pairs) in enumerate(sorted(self.tasks.items())):
            sorted_pairs = sorted(pairs, key=lambda p: p[0])
            n = base + (1 if i < remainder else 0)
            if n > len(sorted_pairs):
                selected = sorted_pairs
                leftover.extend([p for p in sorted_pairs if p not in selected])
            else:
                selected = rng.sample(sorted_pairs, n)
                leftover.extend([p for p in sorted_pairs if p not in selected])
            sampled.extend(selected)
            sampled_counter[task] = len(selected)
        
        # 如果总数不足 x，从 leftover 补齐
        if len(sampled) < x:
            need = x - len(sampled)
            sampled.extend(rng.sample(leftover, need))
        
        rng.shuffle(sampled)
        
        # 打印每类实际采样数量
        print("每类实际采样数量：")
        for task, count in sampled_counter.items():
            print(f"  {task}: {count}")
        
        return sampled
    
    def get_all_prompts(self):
        """获取全部 (idx, prompt)"""
        all_pairs = []
        for pairs in self.tasks.values():
            all_pairs.extend(pairs)
        return all_pairs


if __name__ == "__main__":
    generator = DrawBenchPromptGenerator()
    
    # 返回全部 prompt
    all_prompts = generator.sample_prompts()
    print(f"总共 {len(all_prompts)} 条 prompt")
    
    # 抽样 22 条，可重复
    sampled_prompts = generator.sample_prompts(x=22, seed=42)
    for idx, prompt in sampled_prompts:
        print(idx, prompt)




