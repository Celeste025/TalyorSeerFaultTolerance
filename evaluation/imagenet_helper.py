# download_imagenet_simple.py
import urllib.request

def download_imagenet_classes():
    """下载并保存ImageNet类别"""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    
    try:
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
        
        with open("imagenet_classes.txt", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        print(f"总类别数: {len(classes)}")
        print("下载完成！文件保存为 'imagenet_classes.txt'")
        
        return classes
        
    except Exception as e:
        print(f"下载失败: {e}")
        return None

import os
class ImageNetPromptGenerator:
    def __init__(self, class_file_path="/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/evaluation/datasets/imagenet/imagenet_classes.txt"):
        """
        初始化ImageNet提示词生成器
        
        Args:
            class_file_path: ImageNet类别文件路径
        """
        self.class_file_path = class_file_path
        self.classes = self._load_imagenet_classes()
        
    def _load_imagenet_classes(self):
        """加载ImageNet类别文件"""
        if not os.path.exists(self.class_file_path):
            raise FileNotFoundError(f"ImageNet类别文件不存在: {self.class_file_path}")
        
        with open(self.class_file_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # 验证类别数量
        if len(classes) != 1000:
            print(f"警告: 期望1000个类别，但找到{len(classes)}个类别")
            
        return classes
    
    def get_prompt(self, class_id, template="a photo of a {}."):
        """
        根据类别ID获取提示词
        
        Args:
            class_id: 类别ID (0-999)
            template: 提示词模板，默认"a photo of a {}."
        
        Returns:
            str: 组装好的提示词
        """
        if class_id < 0 or class_id >= len(self.classes):
            raise ValueError(f"类别ID {class_id} 超出范围 (0-{len(self.classes)-1})")
        
        class_name = self.classes[class_id]
        
        # 提取主要的英文类别名称（去除拉丁学名等）
        # 例如: "tench, Tinca tinca" -> "tench"
        primary_name = class_name.split(',')[0].strip()
        
        return template.format(primary_name)
    
    def get_multiple_prompts(self, class_ids, template="a photo of a {}."):
        """
        批量获取多个类别ID的提示词
        
        Args:
            class_ids: 类别ID列表
            template: 提示词模板
        
        Returns:
            list: 提示词列表
        """
        return [self.get_prompt(class_id, template) for class_id in class_ids]
    
    def get_class_name(self, class_id):
        """
        获取原始的类别名称（不组装成prompt）
        
        Args:
            class_id: 类别ID
        
        Returns:
            str: 原始类别名称
        """
        if class_id < 0 or class_id >= len(self.classes):
            raise ValueError(f"类别ID {class_id} 超出范围 (0-{len(self.classes)-1})")
        
        return self.classes[class_id]

if __name__ == "__main__":
    download_imagenet_classes()
    # i = ImageNetPromptGenerator()
    # print(i.get_prompt(1))