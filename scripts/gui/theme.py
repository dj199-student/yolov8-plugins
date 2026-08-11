"""
GUI 主题管理系统

支持亮色/暗色主题切换，自动应用到所有子组件。
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable

# ==================== 主题颜色定义 ====================

LIGHT_THEME = {
    "name": "light",
    "bg": "#f0f0f0",
    "fg": "#1a1a1a",
    "widget_bg": "#ffffff",
    "accent": "#0078d4",
    "accent_hover": "#106ebe",
    "accent_fg": "#ffffff",
    "border": "#cccccc",
    "separator": "#e0e0e0",
    "log_bg": "#1e1e1e",
    "log_fg": "#d4d4d4",
    "log_insert_bg": "#1e1e1e",
    "progress_bg": "#e0e0e0",
    "progress_fg": "#0078d4",
    "tab_bg": "#e8e8e8",
    "tab_selected_bg": "#ffffff",
    "btn_primary_bg": "#0078d4",
    "btn_primary_fg": "#ffffff",
    "btn_danger_bg": "#d32f2f",
    "btn_danger_fg": "#ffffff",
    "btn_success_bg": "#388e3c",
    "btn_success_fg": "#ffffff",
    "font_family": "Microsoft YaHei",
    "font_mono": "Consolas",
    "canvas_bg": "#e0e0e0",
    "text_bg": "#ffffff",
    "text_fg": "#1a1a1a",
}

DARK_THEME = {
    "name": "dark",
    "bg": "#1e1e1e",
    "fg": "#d4d4d4",
    "widget_bg": "#2d2d2d",
    "accent": "#007acc",
    "accent_hover": "#1a8cff",
    "accent_fg": "#ffffff",
    "border": "#3e3e3e",
    "separator": "#333333",
    "log_bg": "#1a1a1a",
    "log_fg": "#cccccc",
    "log_insert_bg": "#1a1a1a",
    "progress_bg": "#3e3e3e",
    "progress_fg": "#007acc",
    "tab_bg": "#2a2a2a",
    "tab_selected_bg": "#1e1e1e",
    "btn_primary_bg": "#007acc",
    "btn_primary_fg": "#ffffff",
    "btn_danger_bg": "#e53935",
    "btn_danger_fg": "#ffffff",
    "btn_success_bg": "#43a047",
    "btn_success_fg": "#ffffff",
    "font_family": "Microsoft YaHei",
    "font_mono": "Consolas",
    "canvas_bg": "#2a2a2a",
    "text_bg": "#2d2d2d",
    "text_fg": "#d4d4d4",
}


class ThemeManager:
    """主题管理器 — 管理主题切换和样式更新"""

    def __init__(self):
        self._theme = LIGHT_THEME.copy()
        self._listeners: list[Callable] = []

    @property
    def current(self) -> dict:
        return self._theme

    @property
    def name(self) -> str:
        return self._theme["name"]

    def get(self, key: str, default=None):
        return self._theme.get(key, default)

    def toggle(self) -> str:
        """切换亮/暗主题，返回新主题名"""
        new_name = "dark" if self._theme["name"] == "light" else "light"
        self.apply(new_name)
        return new_name

    def apply(self, name: str) -> None:
        """应用指定主题"""
        themes = {"light": LIGHT_THEME, "dark": DARK_THEME}
        if name not in themes:
            return
        self._theme = themes[name].copy()
        self._apply_ttk_style(name)
        self._notify_listeners()

    def on_change(self, callback: Callable) -> None:
        """注册主题变更回调"""
        self._listeners.append(callback)

    def _notify_listeners(self) -> None:
        """通知所有监听器"""
        for cb in self._listeners:
            try:
                cb(self._theme)
            except Exception:
                pass

    def _apply_ttk_style(self, name: str) -> None:
        """应用 ttk 样式"""
        style = ttk.Style()
        ttk_theme = "clam"
        style.theme_use(ttk_theme)

        t = self._theme
        font = t["font_family"]
        mono = t["font_mono"]

        # 通用样式
        style.configure("TLabel", font=(font, 9), background=t["bg"], foreground=t["fg"])
        style.configure("TButton", font=(font, 9))
        style.configure("TFrame", background=t["bg"])
        style.configure("TLabelframe", background=t["bg"])
        style.configure("TLabelframe.Label", font=(font, 10, "bold"),
                        background=t["bg"], foreground=t["fg"])

        # 标题
        style.configure("Title.TLabel", font=(font, 18, "bold"),
                        background=t["bg"], foreground=t["fg"])
        style.configure("Header.TLabel", font=(font, 12, "bold"),
                        background=t["bg"], foreground=t["fg"])
        style.configure("Status.TLabel", font=(font, 9),
                        background=t["bg"], foreground=t["fg"])

        # 主按钮
        style.configure("Primary.TButton", font=(font, 10, "bold"))

        # TNotebook
        style.configure("TNotebook", background=t["bg"])
        style.configure("TNotebook.Tab", background=t["tab_bg"], foreground=t["fg"])

        # Progressbar
        style.configure("TProgressbar", background=t["progress_fg"],
                        troughcolor=t["progress_bg"])

        # PanedWindow
        style.configure("TPanedwindow", background=t["bg"])

        # Separator
        style.configure("TSeparator", background=t["separator"])

    def apply_to_tk_widget(self, widget: tk.Widget) -> None:
        """递归地将主题应用到 tk (非 ttk) 组件树"""
        t = self._theme
        try:
            wclass = widget.winfo_class()
            if wclass in ("Text",):
                widget.configure(bg=t["text_bg"], fg=t["text_fg"],
                                insertbackground=t["fg"])
            elif wclass in ("Canvas",):
                widget.configure(bg=t["canvas_bg"])
            elif wclass in ("Label", "Button"):
                widget.configure(bg=t["widget_bg"], fg=t["fg"])
            elif wclass in ("Frame", "Toplevel", "Tk"):
                widget.configure(bg=t["bg"])
        except Exception:
            pass

        # 递归子组件
        for child in widget.winfo_children():
            self.apply_to_tk_widget(child)


# 全局单例
_theme_manager = ThemeManager()


def get_theme_manager() -> ThemeManager:
    return _theme_manager


def get_theme() -> dict:
    return _theme_manager.current


def set_theme(name: str) -> None:
    _theme_manager.apply(name)


def toggle_theme() -> str:
    return _theme_manager.toggle()


# 兼容旧代码
CURRENT_THEME = _theme_manager.current
LIGHT_THEME_DICT = LIGHT_THEME
DARK_THEME_DICT = DARK_THEME
