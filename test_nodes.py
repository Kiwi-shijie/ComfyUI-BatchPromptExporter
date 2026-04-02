"""
ComfyUI-BatchPromptExporter 完整测试
模拟 ComfyUI 实际调用方式验证节点
重点测试 INPUT_IS_LIST=True 和文件名一一对应
"""

import sys
import os
import tempfile
import shutil

plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plugin_dir)


def test_comfyui_loading():
    """测试 ComfyUI 加载"""
    print("=" * 50)
    print("TEST 1: ComfyUI 加载")
    print("=" * 50)

    import importlib
    mod = importlib.import_module("__init__")
    mappings = mod.NODE_CLASS_MAPPINGS

    assert "BatchImageVideoLoader" in mappings
    assert "BatchTextExporter" in mappings

    for name, cls in mappings.items():
        assert hasattr(cls, "INPUT_TYPES")
        assert hasattr(cls, "RETURN_TYPES")
        assert hasattr(cls, "FUNCTION")
        func_name = cls.FUNCTION
        assert hasattr(cls, func_name), f"{name} 缺少方法 {func_name}"
        print(f"  {name}: OK")

    # 关键检查：BatchTextExporter 必须有 INPUT_IS_LIST = True
    exporter_cls = mappings["BatchTextExporter"]
    assert hasattr(exporter_cls, "INPUT_IS_LIST"), "BatchTextExporter 缺少 INPUT_IS_LIST"
    assert exporter_cls.INPUT_IS_LIST is True, "INPUT_IS_LIST 必须为 True"
    print(f"  BatchTextExporter.INPUT_IS_LIST = True: OK")

    print("  PASS\n")
    return True


def test_loader():
    """测试加载图像"""
    print("=" * 50)
    print("TEST 2: 加载图像")
    print("=" * 50)

    from __init__ import BatchImageVideoLoader
    from PIL import Image
    import numpy as np
    import torch

    loader = BatchImageVideoLoader()
    tmpdir = tempfile.mkdtemp(prefix="bp_test_")

    try:
        # 创建测试图像，模拟用户实际文件名（包含空格和括号）
        test_names = ["1 (1)", "1 (2)", "1 (3)", "1 (4)", "1 (5)"]
        for name in test_names:
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
            )
            img.save(os.path.join(tmpdir, f"{name}.jpg"))

        images, masks, filenames, count = loader.load(tmpdir)

        assert count == 5, f"应加载 5 个，实际 {count}"
        assert len(images) == 5
        assert len(filenames) == 5

        # 验证文件名保留了空格和括号
        for fn in filenames:
            assert fn in test_names, f"文件名不匹配: '{fn}'"

        # 验证 tensor 格式
        for img_t in images:
            assert img_t.shape[0] == 1, "batch dim = 1"
            assert img_t.shape[3] == 3, "channels = 3"
            assert img_t.dtype == torch.float32

        print(f"  加载 {count} 项: {filenames}")
        print(f"  tensor shape: {images[0].shape}")

    finally:
        shutil.rmtree(tmpdir)

    print("  PASS\n")
    return True


def test_loader_errors():
    """测试加载器错误处理"""
    print("=" * 50)
    print("TEST 3: 加载器错误处理")
    print("=" * 50)

    from __init__ import BatchImageVideoLoader
    loader = BatchImageVideoLoader()

    for label, path in [("空路径", ""), ("无效路径", "C:/no/such/path/999")]:
        try:
            loader.load(path)
            assert False
        except ValueError:
            print(f"  {label}: ValueError OK")

    tmpdir = tempfile.mkdtemp(prefix="bp_empty_")
    try:
        loader.load(tmpdir)
        assert False
    except ValueError:
        print(f"  空目录: ValueError OK")
    finally:
        shutil.rmtree(tmpdir)

    print("  PASS\n")
    return True


def test_exporter_input_is_list():
    """
    测试导出器在 INPUT_IS_LIST=True 模式下的行为。
    ComfyUI 会把所有参数都包装成列表传入。
    """
    print("=" * 50)
    print("TEST 4: 导出器 INPUT_IS_LIST 模式")
    print("=" * 50)

    from __init__ import BatchTextExporter
    exporter = BatchTextExporter()

    tmpdir = tempfile.mkdtemp(prefix="bp_export_")
    try:
        texts = [
            "1girl, underwear, sitting",
            "1girl, dress, standing",
            "1girl, swimsuit, beach",
            "close-up, face, smile",
            "full body, outdoor, sunset",
        ]
        filenames = ["1 (1)", "1 (2)", "1 (3)", "1 (4)", "1 (5)"]

        # ComfyUI INPUT_IS_LIST=True 时，所有参数都是列表
        report, saved = exporter.export(
            texts=texts,
            filenames=filenames,
            output_folder=[tmpdir],
            overwrite=[True],
            encoding=["utf-8"],
        )

        assert saved == 5, f"应保存 5 个，实际 {saved}"

        # 验证每个文件
        for i, name in enumerate(filenames):
            txt_path = os.path.join(tmpdir, f"{name}.txt")
            assert os.path.isfile(txt_path), f"缺少: {txt_path}"
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == texts[i], f"内容不匹配"

        # 关键：没有多余文件
        txt_files = [f for f in os.listdir(tmpdir) if f.endswith(".txt")]
        assert len(txt_files) == 5, f"应只有 5 个 txt，实际: {txt_files}"

        print(f"  保存 {saved} 个, 无多余文件: OK")

    finally:
        shutil.rmtree(tmpdir)

    print("  PASS\n")
    return True


def test_no_extra_files():
    """
    核心测试：texts 多于 filenames 时，不产生 item_XXXX 等多余文件。
    这是之前 bug 的根源。
    """
    print("=" * 50)
    print("TEST 5: texts > filenames 时无多余文件")
    print("=" * 50)

    from __init__ import BatchTextExporter
    exporter = BatchTextExporter()

    tmpdir = tempfile.mkdtemp(prefix="bp_mismatch_")
    try:
        # 9 条文本但只有 5 个文件名
        texts = ["t1", "t2", "t3", "t4", "t5", "extra1", "extra2", "extra3", "extra4"]
        filenames = ["1 (1)", "1 (2)", "1 (3)", "1 (4)", "1 (5)"]

        report, saved = exporter.export(
            texts=texts, filenames=filenames,
            output_folder=[tmpdir], overwrite=[True], encoding=["utf-8"],
        )

        assert saved == 5, f"应只保存 5 个，实际 {saved}"

        all_files = os.listdir(tmpdir)
        assert len(all_files) == 5, f"应只有 5 个文件，实际: {all_files}"

        for f in all_files:
            assert "item_" not in f, f"不应有 item_ 文件: {f}"

        print(f"  9 texts + 5 filenames -> 只导出 5 个: OK")
        print(f"  无 item_XXXX: OK")

    finally:
        shutil.rmtree(tmpdir)

    print("  PASS\n")
    return True


def test_full_pipeline():
    """完整流程: 加载 -> 模拟打标 -> 导出，验证一一对应"""
    print("=" * 50)
    print("TEST 6: 完整流程")
    print("=" * 50)

    from __init__ import BatchImageVideoLoader, BatchTextExporter
    from PIL import Image
    import numpy as np

    loader = BatchImageVideoLoader()
    exporter = BatchTextExporter()

    img_dir = tempfile.mkdtemp(prefix="bp_src_")
    out_dir = tempfile.mkdtemp(prefix="bp_out_")

    try:
        names = ["1 (1)", "1 (2)", "1 (3)", "1 (4)", "1 (5)"]
        for name in names:
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(os.path.join(img_dir, f"{name}.jpg"))

        # 加载
        images, masks, filenames, count = loader.load(img_dir)
        print(f"  加载: {count} 项")

        # 模拟打标
        tags = [f"tag_for_{fn}" for fn in filenames]

        # 导出（模拟 ComfyUI INPUT_IS_LIST 传参）
        report, saved = exporter.export(
            texts=tags, filenames=filenames,
            output_folder=[out_dir], overwrite=[True], encoding=["utf-8"],
        )
        assert saved == count

        # 验证目录里只有 count 个 txt 文件
        all_out = os.listdir(out_dir)
        assert len(all_out) == count, f"输出文件数不对: {all_out}"

        for i, fn in enumerate(filenames):
            txt_path = os.path.join(out_dir, f"{fn}.txt")
            assert os.path.isfile(txt_path)
            with open(txt_path, "r") as f:
                assert f.read() == tags[i]
            print(f"    {fn}.jpg <-> {fn}.txt OK")

    finally:
        shutil.rmtree(img_dir)
        shutil.rmtree(out_dir)

    print("  PASS\n")
    return True


if __name__ == "__main__":
    import torch

    tests = [
        ("ComfyUI加载", test_comfyui_loading),
        ("加载图像", test_loader),
        ("错误处理", test_loader_errors),
        ("INPUT_IS_LIST导出", test_exporter_input_is_list),
        ("无多余文件", test_no_extra_files),
        ("完整流程", test_full_pipeline),
    ]

    results = []
    for name, func in tests:
        try:
            results.append((name, func()))
        except Exception:
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("=" * 50)
    print("总结")
    print("=" * 50)
    all_pass = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print("=" * 50)
    print("全部通过!" if all_pass else "有测试失败!")
    sys.exit(0 if all_pass else 1)
