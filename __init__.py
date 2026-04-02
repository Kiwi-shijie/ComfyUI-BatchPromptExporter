"""
ComfyUI-BatchPromptExporter
两个核心节点：
  1. BatchImageVideoLoader - 从文件夹批量加载图像/视频，输出 IMAGE 列表供下游使用
  2. BatchTextExporter     - 将打标结果按原文件名批量导出为 .txt 文件
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps

# opencv 用于视频，可选依赖
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ============================================================
# 工具函数
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv"}


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image -> ComfyUI IMAGE tensor (1, H, W, 3) float32 [0,1]"""
    img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋转
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)


def _pil_to_mask(img: Image.Image) -> torch.Tensor:
    """提取 alpha 通道作为 mask，无 alpha 则全白"""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.getchannel("A"), dtype=np.float32) / 255.0
    else:
        w, h = img.size
        alpha = np.ones((h, w), dtype=np.float32)
    return torch.from_numpy(alpha).unsqueeze(0)  # (1, H, W)


# ============================================================
# 节点 1：批量加载图像/视频
# ============================================================

class BatchImageVideoLoader:
    """
    从指定文件夹批量加载图像和视频文件。
    视频会按设定的帧间隔提取帧。
    输出标准 IMAGE 列表，可直接连接到任何下游节点。
    同时输出 filenames 列表，传给 BatchTextExporter 做同名导出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "placeholder": "C:/path/to/images",
                }),
            },
            "optional": {
                "load_images": ("BOOLEAN", {"default": True}),
                "load_videos": ("BOOLEAN", {"default": True}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "sort_by": (["filename", "modified_time", "file_size"], {"default": "filename"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
                "video_frame_interval": ("INT", {
                    "default": 10, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "每隔 N 帧抽取 1 帧，1 = 取全部帧",
                }),
                "video_max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "每个视频最多取多少帧，0 = 不限制",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("images", "masks", "filenames", "total_count")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "load"
    CATEGORY = "BatchPrompt"

    def load(self, folder_path, load_images=True, load_videos=True,
             include_subfolders=False, sort_by="filename", reverse_order=False,
             video_frame_interval=10, video_max_frames=0):

        folder_path = folder_path.strip()
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"文件夹路径无效: '{folder_path}'")

        # 1. 收集文件
        all_files = self._collect_files(folder_path, include_subfolders,
                                         load_images, load_videos)
        if not all_files:
            raise ValueError(f"文件夹中没有找到符合条件的图像/视频文件: '{folder_path}'")

        # 2. 排序
        all_files = self._sort_files(all_files, sort_by, reverse_order)

        # 3. 加载
        images = []
        masks = []
        filenames = []

        for fpath in all_files:
            ext = os.path.splitext(fpath)[1].lower()

            if ext in IMAGE_EXTS:
                try:
                    pil_img = Image.open(fpath)
                    images.append(_pil_to_tensor(pil_img))
                    masks.append(_pil_to_mask(pil_img))
                    # 记录不含扩展名的文件名
                    filenames.append(os.path.splitext(os.path.basename(fpath))[0])
                except Exception as e:
                    print(f"[BatchPrompt] 跳过损坏图像 {fpath}: {e}")

            elif ext in VIDEO_EXTS:
                if not _HAS_CV2:
                    print(f"[BatchPrompt] 缺少 opencv-python，跳过视频: {fpath}")
                    continue
                self._load_video_frames(
                    fpath, images, masks, filenames,
                    video_frame_interval, video_max_frames
                )

        if not images:
            raise ValueError("没有成功加载任何图像或视频帧")

        print(f"[BatchPrompt] 共加载 {len(images)} 项 "
              f"(来自 {len(all_files)} 个文件)")

        return (images, masks, filenames, len(images))

    # ------ 内部方法 ------

    def _collect_files(self, folder, recursive, want_images, want_videos):
        allowed = set()
        if want_images:
            allowed |= IMAGE_EXTS
        if want_videos and _HAS_CV2:
            allowed |= VIDEO_EXTS

        results = []
        if recursive:
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in allowed:
                        results.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(folder):
                fp = os.path.join(folder, fn)
                if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in allowed:
                    results.append(fp)
        return results

    @staticmethod
    def _sort_files(files, sort_by, reverse):
        if sort_by == "modified_time":
            key = os.path.getmtime
        elif sort_by == "file_size":
            key = os.path.getsize
        else:
            key = lambda p: os.path.basename(p).lower()
        return sorted(files, key=key, reverse=reverse)

    @staticmethod
    def _load_video_frames(video_path, images, masks, filenames,
                           interval, max_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[BatchPrompt] 无法打开视频: {video_path}")
            return

        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(1, interval) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                arr = rgb.astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0)
                mask = torch.ones(1, arr.shape[0], arr.shape[1], dtype=torch.float32)

                images.append(tensor)
                masks.append(mask)
                filenames.append(f"{video_basename}_frame{frame_idx:06d}")
                extracted += 1

                if max_frames > 0 and extracted >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        print(f"[BatchPrompt] 视频 {video_basename}: "
              f"提取 {extracted} 帧 (共 {frame_idx} 帧)")


# ============================================================
# 节点 2：批量导出 .txt
# ============================================================

class BatchTextExporter:
    """
    将标注/提示词批量导出为 .txt 文件。
    每个文件名与原图/视频帧严格一一对应（通过 filenames 输入）。
    声明 INPUT_IS_LIST 以接收完整列表，避免 ComfyUI 逐条展开导致错乱。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {"forceInput": True}),
                "filenames": ("STRING", {"forceInput": True}),
                "output_folder": ("STRING", {
                    "default": "",
                    "placeholder": "C:/path/to/output",
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": True}),
                "encoding": (["utf-8", "utf-8-sig", "gbk"], {"default": "utf-8"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "saved_count")
    OUTPUT_IS_LIST = (False, False)
    # 关键：声明接收列表输入，防止 ComfyUI 逐条展开
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "export"
    CATEGORY = "BatchPrompt"

    def export(self, texts, filenames, output_folder,
               overwrite=None, encoding=None):
        """
        因为 INPUT_IS_LIST = True，所有参数都会以列表形式传入。
        需要自己解包非列表参数。
        """

        # ---- 解包参数（INPUT_IS_LIST=True 时所有参数都是 list）----
        # output_folder: 只取第一个值
        if isinstance(output_folder, list):
            output_folder = output_folder[0]
        output_folder = output_folder.strip()

        # overwrite: 只取第一个值
        if isinstance(overwrite, list):
            overwrite = overwrite[0]
        if overwrite is None:
            overwrite = True

        # encoding: 只取第一个值
        if isinstance(encoding, list):
            encoding = encoding[0]
        if encoding is None:
            encoding = "utf-8"

        # texts 和 filenames 保持为列表
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(filenames, str):
            filenames = [filenames]

        # ---- 核心逻辑 ----
        if not output_folder:
            raise ValueError("请指定输出文件夹路径")

        os.makedirs(output_folder, exist_ok=True)

        # 严格按 filenames 数量导出，不生成多余文件
        count = min(len(texts), len(filenames))
        if count == 0:
            return ("没有可导出的内容", 0)

        saved = 0
        skipped = 0
        errors = 0
        error_lines = []

        for i in range(count):
            name = filenames[i]
            text = texts[i]
            txt_path = os.path.join(output_folder, f"{name}.txt")

            # 是否跳过已存在的文件
            if not overwrite and os.path.exists(txt_path):
                skipped += 1
                continue

            try:
                with open(txt_path, "w", encoding=encoding) as f:
                    f.write(text)
                saved += 1
            except Exception as e:
                errors += 1
                error_lines.append(f"  {name}: {e}")

        # 如果 texts 比 filenames 多，提示被忽略
        ignored = len(texts) - count

        report = f"导出完成: {saved} 保存, {skipped} 跳过, {errors} 失败"
        if ignored > 0:
            report += f", {ignored} 忽略(无对应文件名)"
        report += f"\n输出目录: {output_folder}"
        if error_lines:
            report += "\n错误:\n" + "\n".join(error_lines)

        print(f"[BatchPrompt] {report}")
        return (report, saved)


# ============================================================
# ComfyUI 注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "BatchImageVideoLoader": BatchImageVideoLoader,
    "BatchTextExporter": BatchTextExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageVideoLoader": "Batch Load Images/Videos | 批量加载",
    "BatchTextExporter": "Batch Export TXT | 批量导出标注",
}
