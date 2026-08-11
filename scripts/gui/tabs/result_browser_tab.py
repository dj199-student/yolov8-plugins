"""
结果浏览器标签页

统一浏览和管理所有输出结果：
- 训练输出 (runs/detect/train*/)
- 检测结果 (detect_output/)
- 验证结果 (runs/detect/val*/)
- 导出文件 (.onnx, .engine, .tflite, ...)
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ..config import load_config, save_config


class ResultBrowserTab:
    """结果浏览器标签页"""

    SCAN_DIRS = [
        "runs",
        "detect_output",
        "exports",
    ]

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        self._build_ui()
        self._refresh()

    # ==================== UI ====================

    def _build_ui(self) -> None:
        """构建结果浏览器 UI"""
        # 工具栏
        toolbar = ttk.Frame(self.parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="🔄 刷新", command=self._refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 打开选中", command=self._open_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗑 删除选中", command=self._delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 全部打开", command=self._open_all).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="  |  ").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="📁 打开根目录", command=self._open_root).pack(side=tk.LEFT, padx=2)

        # 主内容区
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- 左侧：目录树 ----
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        ttk.Label(left, text="📁 目录", font=("Microsoft YaHei", 10, "bold")).pack(
            anchor=tk.W, padx=5, pady=(5, 0))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        tree_scroll = ttk.Scrollbar(tree_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # ---- 右侧：文件列表 + 预览 ----
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # 文件列表
        list_label_frame = ttk.LabelFrame(right, text="📄 文件列表", padding=5)
        list_label_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "size", "modified")
        self.file_list = ttk.Treeview(
            list_label_frame, columns=columns, show="headings", selectmode="extended",
        )
        self.file_list.heading("name", text="文件名")
        self.file_list.heading("size", text="大小")
        self.file_list.heading("modified", text="修改时间")
        self.file_list.column("name", width=250)
        self.file_list.column("size", width=80)
        self.file_list.column("modified", width=140)

        file_scroll = ttk.Scrollbar(list_label_frame, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 双击打开
        self.file_list.bind("<Double-1>", lambda e: self._open_selected_file())
        self.file_list.bind("<Delete>", lambda e: self._delete_selected_file())

        # 底部：选中文件信息
        self.info_label = ttk.Label(right, text="选择文件查看详情", font=("Consolas", 9))
        self.info_label.pack(fill=tk.X, padx=5, pady=3)

    # ==================== 扫描逻辑 ====================

    def _refresh(self) -> None:
        """刷新目录树"""
        self.tree.delete(*self.tree.get_children())
        self.file_list.delete(*self.file_list.get_children())

        cwd = Path.cwd()
        roots = []

        for scan_dir in self.SCAN_DIRS:
            p = cwd / scan_dir
            if p.exists() and p.is_dir():
                roots.append(p)

        if not roots:
            self.tree.insert("", tk.END, text="(没有找到结果目录)", values=("",))
            return

        for root in roots:
            root_id = self.tree.insert("", tk.END, text=root.name, values=(str(root),), open=True)
            self._populate_tree(root_id, root, depth=0)

        self.status_var.set(f"结果浏览器 — {len(roots)} 个根目录")

    def _populate_tree(self, parent_id: str, path: Path, depth: int) -> None:
        """递归填充目录树（最多 3 层）"""
        if depth >= 3:
            return
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            dirs = [p for p in items if p.is_dir()]
            # 跳过 __pycache__ 等隐藏目录
            dirs = [d for d in dirs if not d.name.startswith(".") and d.name != "__pycache__"]
            for d in dirs:
                child_id = self.tree.insert(parent_id, tk.END, text=d.name, values=(str(d),))
                self._populate_tree(child_id, d, depth + 1)
        except PermissionError:
            pass

    def _on_tree_select(self, event) -> None:
        """目录树选择回调 — 显示该目录下的文件"""
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        if not values or not values[0]:
            return
        dir_path = Path(values[0])
        self._list_files(dir_path)

    def _list_files(self, dir_path: Path) -> None:
        """列出目录中的文件"""
        self.file_list.delete(*self.file_list.get_children())
        if not dir_path.exists():
            return

        try:
            files = sorted(
                [p for p in dir_path.iterdir() if p.is_file()],
                key=lambda x: x.stat().st_mtime, reverse=True,
            )
            for f in files:
                stat = f.stat()
                size = stat.st_size
                if size > 1e6:
                    size_str = f"{size / 1e6:.1f} MB"
                elif size > 1e3:
                    size_str = f"{size / 1e3:.1f} KB"
                else:
                    size_str = f"{size} B"

                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                self.file_list.insert("", tk.END, values=(f.name, size_str, mtime), tags=(str(f),))

            self.info_label.configure(text=f"{dir_path} — {len(files)} 个文件")
        except PermissionError:
            self.info_label.configure(text=f"{dir_path} — 无权限访问")

    # ==================== 操作 ====================

    def _open_selected(self) -> None:
        """打开选中的目录"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先在左侧目录树中选择一个目录")
            return
        values = self.tree.item(selection[0], "values")
        if values and values[0]:
            os.startfile(values[0])

    def _open_selected_file(self) -> None:
        """打开选中的文件"""
        selection = self.file_list.selection()
        if not selection:
            return
        item = selection[0]
        tags = self.file_list.item(item, "tags")
        if tags and tags[0]:
            os.startfile(tags[0])

    def _delete_selected_file(self) -> None:
        """删除选中的文件"""
        selection = self.file_list.selection()
        if not selection:
            return
        if not messagebox.askyesno("确认", f"确定要删除 {len(selection)} 个文件吗？"):
            return
        for item in selection:
            tags = self.file_list.item(item, "tags")
            if tags and tags[0]:
                try:
                    os.remove(tags[0])
                    self.file_list.delete(item)
                except OSError as e:
                    messagebox.showerror("错误", f"删除失败: {e}")
        self.status_var.set(f"已删除 {len(selection)} 个文件")

    def _delete_selected(self) -> None:
        """删除选中的目录"""
        selection = self.tree.selection()
        if not selection:
            return
        item = selection[0]
        values = self.tree.item(item, "values")
        if not values or not values[0]:
            return
        dir_path = values[0]
        if messagebox.askyesno("确认", f"确定要删除目录？\n{dir_path}"):
            try:
                import shutil
                shutil.rmtree(dir_path)
                self.tree.delete(item)
                self.file_list.delete(*self.file_list.get_children())
                self.status_var.set(f"已删除: {Path(dir_path).name}")
            except OSError as e:
                messagebox.showerror("错误", f"删除失败: {e}")

    def _open_all(self) -> None:
        """打开所有根目录"""
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            if values and values[0]:
                os.startfile(values[0])

    def _open_root(self) -> None:
        """打开项目根目录"""
        os.startfile(str(Path.cwd()))

    # ==================== 配置 ====================

    def save_config(self) -> None:
        pass  # 结果浏览器无配置需要保存
