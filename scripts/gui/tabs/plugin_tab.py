"""
插件可视化管理标签页

功能：
- 浏览 35+ 可插拔模块（6 大类别）
- 搜索/过滤插件
- 查看插件详情（名称、参数、描述）
- 构建模型配置（backbone / neck / head）
- 导出 YAML 配置文件
- 后台构建带插件的模型
"""

import os
import sys
import inspect
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from models.registry import PLUGIN_REGISTRY, list_plugins
from ..widgets.log_panel import LogPanel
from ..workers.plugin_worker import PluginWorker
from ..config import load_config, save_config


class PluginTab:
    """插件可视化管理标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 当前模型配置
        self._model_config = {
            "base": "yolov8n.pt",
            "plugins": {
                "backbone": [],
                "neck": [],
                "head": [],
            },
        }

        # 当前选中的插件
        self._selected_plugin: str | None = None

        # Worker
        self.worker = PluginWorker()
        self.worker.on_log(self._on_log)
        self.worker.on_done(self._on_build_done)
        self.worker.on_error(self._on_build_error)

        self._build_ui()

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建插件标签页 UI

        左: 搜索 + 分类树 + 基础模型选择
        右: 插件详情 + 模型配置
        底: 日志
        """
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧 ----
        left_container = ttk.Frame(paned, width=300)
        paned.add(left_container, weight=0)

        left_canvas = tk.Canvas(left_container, width=280, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor=tk.NW, tags="inner")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        def _configure_width(event):
            left_canvas.itemconfig("inner", width=event.width)
        left_canvas.bind("<Configure>", _configure_width, add="+")

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 基础模型选择
        base_frame = ttk.LabelFrame(left, text="基础模型", padding=8)
        base_frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        self.base_var = tk.StringVar(value="yolov8n.pt")
        ttk.Label(base_frame, text="基础权重:").pack(anchor=tk.W)
        base_combo = ttk.Combobox(
            base_frame, textvariable=self.base_var,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            state="readonly", width=20,
        )
        base_combo.pack(fill=tk.X, pady=2)
        base_combo.bind("<<ComboboxSelected>>", lambda e: self._on_base_changed())

        # 搜索框
        search_frame = ttk.LabelFrame(left, text="搜索", padding=5)
        search_frame.pack(fill=tk.X, padx=5, pady=3)

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X)
        search_entry.bind("<KeyRelease>", lambda e: self._on_search())

        ttk.Label(search_frame, text="输入关键词过滤插件",
                  font=("Microsoft YaHei", 8)).pack(anchor=tk.W)

        # 分类树
        tree_frame = ttk.LabelFrame(left, text="插件分类", padding=5)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        self._tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse", height=20)
        self._tree.pack(fill=tk.BOTH, expand=True)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.configure(yscrollcommand=tree_scroll.set)

        self._tree.bind("<<TreeviewSelect>>", self._on_plugin_selected)

        # 加载插件数据
        self._all_plugins: dict[str, list[str]] = {}
        self._populate_tree()

        # ---- 右侧 ----
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        v_paned = ttk.PanedWindow(right, orient=tk.VERTICAL)
        v_paned.pack(fill=tk.BOTH, expand=True)

        # 上半：插件详情 + 模型配置
        top_right = ttk.Frame(v_paned)
        v_paned.add(top_right, weight=1)

        # 插件详情面板
        detail_frame = ttk.LabelFrame(top_right, text="插件详情", padding=10)
        detail_frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        self._detail_text = tk.Text(
            detail_frame, height=10, wrap=tk.WORD,
            font=("Consolas", 10), state=tk.DISABLED,
            bg="#fafafa", fg="#1a1a1a",
            relief=tk.FLAT, borderwidth=0,
        )
        self._detail_text.pack(fill=tk.BOTH, expand=True)

        # 添加到模型按钮
        detail_btn_row = ttk.Frame(detail_frame)
        detail_btn_row.pack(fill=tk.X, pady=(5, 0))

        # 注入位置选择
        ttk.Label(detail_btn_row, text="注入到:").pack(side=tk.LEFT, padx=(0, 5))
        self._inject_target = tk.StringVar(value="backbone")
        for label, val in [("Backbone", "backbone"), ("Neck", "neck"), ("Head", "head")]:
            ttk.Radiobutton(
                detail_btn_row, text=label, variable=self._inject_target,
                value=val,
            ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            detail_btn_row, text="🔧 添加到模型配置",
            command=self._add_to_config,
            style="Primary.TButton",
        ).pack(side=tk.RIGHT, padx=5)

        # 模型配置面板
        config_frame = ttk.LabelFrame(top_right, text="当前模型配置", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        self._config_text = tk.Text(
            config_frame, height=8, wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#fafafa", fg="#1a1a1a",
        )
        self._config_text.pack(fill=tk.BOTH, expand=True)

        # 操作按钮行
        btn_row = ttk.Frame(top_right)
        btn_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_row, text="🎯 构建模型",
                   command=self._build_model,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="💾 导出 YAML",
                   command=self._export_yaml).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="🗑 清空配置",
                   command=self._clear_config).pack(side=tk.LEFT, padx=2)

        # 构建状态
        self._build_status_var = tk.StringVar(value="就绪")
        ttk.Label(btn_row, textvariable=self._build_status_var,
                  font=("Microsoft YaHei", 9)).pack(side=tk.RIGHT, padx=5)

        # 下半：日志
        log_frame = ttk.Frame(v_paned)
        v_paned.add(log_frame, weight=1)

        self.log_panel = LogPanel(log_frame, title="构建日志")
        self.log_panel.pack(fill=tk.BOTH, expand=True)

        self._poll_log()

        # 初始化
        self._refresh_config_display()
        self.parent.after(200, lambda: self._init_sash(paned, v_paned))

    def _init_sash(self, h_paned, v_paned) -> None:
        try:
            w = self.parent.winfo_width()
            if w > 400:
                h_paned.sashpos(0, 300)
            h = self.parent.winfo_height()
            if h > 300:
                v_paned.sashpos(0, int(h * 0.6))
        except Exception:
            pass

    # ==================== 插件树 ====================

    def _populate_tree(self, filter_text: str = "") -> None:
        """填充插件分类树"""
        # 清空
        for item in self._tree.get_children():
            self._tree.delete(item)

        try:
            self._all_plugins = PLUGIN_REGISTRY.list_all()
        except Exception:
            self._all_plugins = {}

        filter_lower = filter_text.lower().strip()

        cat_order = ["attention", "conv", "transformer", "neck", "spp", "head"]
        cat_labels = {
            "attention": "📁 注意力 (Attention)",
            "conv": "📁 卷积 (Convolution)",
            "transformer": "📁 Transformer",
            "neck": "📁 Neck",
            "spp": "📁 SPP",
            "head": "📁 Head",
        }

        for cat in cat_order:
            plugins = self._all_plugins.get(cat, [])
            if not plugins:
                continue

            # 过滤
            if filter_lower:
                plugins = [p for p in plugins if filter_lower in p.lower()]

            if not plugins:
                continue

            cat_label = cat_labels.get(cat, f"📁 {cat}")
            cat_id = self._tree.insert("", "end", text=f"{cat_label} ({len(plugins)})", open=True, values=(cat,))

            for name in sorted(plugins):
                try:
                    meta = PLUGIN_REGISTRY.get_metadata(name)
                    class_name = meta.get("class_name", name)
                    desc = meta.get("description", "")
                    # 截断描述
                    if len(desc) > 60:
                        desc = desc[:57] + "..."
                    display = f"{name}"
                    if desc:
                        display = f"{name}  —  {desc}"
                except Exception:
                    display = name

                self._tree.insert(cat_id, "end", text=display, values=(cat, name))

        # 如果没有匹配，显示提示
        if not self._tree.get_children():
            self._tree.insert("", "end", text="（无匹配插件）")

    def _on_search(self) -> None:
        """搜索过滤"""
        self._populate_tree(self.search_var.get())

    def _on_plugin_selected(self, event) -> None:
        """选中插件时显示详情"""
        selection = self._tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self._tree.item(item, "values")

        # 判断是否是叶子节点（插件）还是分类节点
        if not values or len(values) < 2:
            return

        cat, name = values
        self._selected_plugin = name
        self._show_plugin_detail(name)

    # ==================== 插件详情 ====================

    def _show_plugin_detail(self, name: str) -> None:
        """显示插件详情"""
        try:
            meta = PLUGIN_REGISTRY.get_metadata(name)
            cls = PLUGIN_REGISTRY.get(name)
        except Exception as e:
            self._detail_text.configure(state=tk.NORMAL)
            self._detail_text.delete("1.0", tk.END)
            self._detail_text.insert("1.0", f"加载插件失败: {e}")
            self._detail_text.configure(state=tk.DISABLED)
            return

        category = meta.get("category", "unknown")
        class_name = meta.get("class_name", name)
        description = meta.get("description", "").strip()

        # 解析构造参数
        params_text = ""
        try:
            sig = inspect.signature(cls.__init__)
            for p_name, p in sig.parameters.items():
                if p_name == "self":
                    continue
                if p.default is not inspect.Parameter.empty:
                    params_text += f"  {p_name}: {p.annotation.__name__ if p.annotation != inspect.Parameter.empty else 'Any'} = {p.default!r}\n"
                else:
                    params_text += f"  {p_name} (必需)\n"
        except Exception:
            params_text = "  (无法解析参数)\n"

        if not params_text:
            params_text = "  (无参数)\n"

        # 渲染详情
        lines = [
            f"名称: {name}",
            f"类别: {category}",
            f"类名: {class_name}",
            f"参数:",
            params_text.rstrip(),
            "",
            f"描述:",
            f"  {description or '(无描述)'}",
        ]

        self._detail_text.configure(state=tk.NORMAL)
        self._detail_text.delete("1.0", tk.END)
        self._detail_text.insert("1.0", "\n".join(lines))
        self._detail_text.configure(state=tk.DISABLED)

        self.status_var.set(f"选中插件: {name}")

    # ==================== 模型配置管理 ====================

    def _add_to_config(self) -> None:
        """将选中插件添加到模型配置"""
        if not self._selected_plugin:
            messagebox.showwarning("提示", "请先在左侧选择插件")
            return

        target = self._inject_target.get()
        self._model_config["plugins"][target].append({
            "type": self._selected_plugin,
            "params": {},
        })

        self._refresh_config_display()
        self.status_var.set(f"已添加 {self._selected_plugin} → {target}")

    def _remove_from_config(self, section: str, index: int) -> None:
        """从配置中移除插件"""
        try:
            removed = self._model_config["plugins"][section].pop(index)
            self._refresh_config_display()
            self.status_var.set(f"已移除 {removed['type']}")
        except IndexError:
            pass

    def _clear_config(self) -> None:
        """清空模型配置"""
        if not any(self._model_config["plugins"].values()):
            return

        if messagebox.askyesno("确认", "确定清空所有插件配置？"):
            for section in self._model_config["plugins"]:
                self._model_config["plugins"][section] = []
            self._refresh_config_display()
            self.status_var.set("配置已清空")

    def _on_base_changed(self) -> None:
        """基础模型变更"""
        self._model_config["base"] = self.base_var.get()
        self._refresh_config_display()

    def _refresh_config_display(self) -> None:
        """刷新配置显示"""
        lines = [f"基础模型: {self._model_config['base']}", ""]

        for section in ("backbone", "neck", "head"):
            plugins = self._model_config["plugins"].get(section, [])
            if plugins:
                names = [f"{p['type']}" for p in plugins]
                lines.append(f"{section}:")
                for i, name in enumerate(names):
                    lines.append(f"  [{i}] {name}")
            else:
                lines.append(f"{section}: (原始)")
            lines.append("")

        self._config_text.configure(state=tk.NORMAL)
        self._config_text.delete("1.0", tk.END)
        self._config_text.insert("1.0", "\n".join(lines))
        self._config_text.configure(state=tk.DISABLED)

        # 高亮不同部分
        self._apply_config_tags()

    def _apply_config_tags(self) -> None:
        """为配置文本添加颜色标记"""
        try:
            self._config_text.tag_configure("section", foreground="#0078d4", font=("Consolas", 10, "bold"))
            self._config_text.tag_configure("plugin", foreground="#d32f2f")
            self._config_text.tag_configure("base", foreground="#388e3c", font=("Consolas", 10, "bold"))
        except Exception:
            pass

    # ==================== 构建模型 ====================

    def _build_model(self) -> None:
        """后台构建模型"""
        total_plugins = sum(
            len(v) for v in self._model_config["plugins"].values()
        )
        if total_plugins == 0:
            messagebox.showinfo("提示", "未添加任何插件，将加载原始基础模型")

        self._build_status_var.set("构建中...")
        self.worker.build_model(self._model_config)

    def _on_build_done(self, summary: dict, model) -> None:
        """构建完成"""
        self.parent.after(0, lambda: self._show_build_result(summary))

    def _show_build_result(self, summary: dict) -> None:
        """显示构建结果（主线程）"""
        self._build_status_var.set(
            f"✅ 构建完成 | {summary['total_plugins']} 个插件 | {summary['base_model']}"
        )
        self.status_var.set("模型构建完成")
        messagebox.showinfo(
            "构建完成",
            f"插件模型构建成功！\n\n"
            f"基础模型: {summary['base_model']}\n"
            f"插件数量: {summary['total_plugins']}\n"
        )

    def _on_build_error(self, error: str) -> None:
        """构建错误"""
        self.parent.after(0, lambda: self._handle_build_error(error))

    def _handle_build_error(self, error: str) -> None:
        """处理构建错误（主线程）"""
        self._build_status_var.set("❌ 构建失败")
        self.status_var.set("模型构建失败")
        messagebox.showerror("构建错误", error)

    # ==================== 导出 YAML ====================

    def _export_yaml(self) -> None:
        """导出配置到 YAML 文件"""
        total_plugins = sum(
            len(v) for v in self._model_config["plugins"].values()
        )
        if total_plugins == 0:
            messagebox.showinfo("提示", "未添加任何插件，无法导出空配置")
            return

        path = filedialog.asksaveasfilename(
            title="导出插件配置",
            defaultextension=".yaml",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
            initialfile="plugin_config.yaml",
        )
        if not path:
            return

        try:
            import yaml

            config = {"model": self._model_config}
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            self.status_var.set(f"配置已导出: {Path(path).name}")
            messagebox.showinfo("导出完成", f"插件配置已保存至:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ==================== 日志 ====================

    def _on_log(self, msg: str) -> None:
        self.log_panel.write(msg)

    def _poll_log(self) -> None:
        self.log_panel.poll()
        self.parent.after(200, self._poll_log)

    # ==================== 公共接口 ====================

    def refresh_models(self) -> None:
        """刷新（插件无需刷新，但保持接口一致）"""
        pass

    def save_config(self) -> None:
        """保存插件配置"""
        cfg = load_config()
        cfg.setdefault("plugins", {}).update({
            "base_model": self._model_config["base"],
            "last_plugins": {
                "backbone": [p["type"] for p in self._model_config["plugins"]["backbone"]],
                "neck": [p["type"] for p in self._model_config["plugins"]["neck"]],
                "head": [p["type"] for p in self._model_config["plugins"]["head"]],
            },
        })
        save_config(cfg)
