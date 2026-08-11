# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-08-11

### Added

- **Plugin system** with registry center and YAML-driven configuration.
- **30+ attention modules**: SE, CBAM, ECA, CA, GAM, SimAM, Shuffle Attention, Triplet Attention, Dual Attention.
- **7 improved convolution modules**: GhostConv, DWConv, RepConv, DynamicConv, ODConv, DSConv, PConv, Involution.
- **4 Transformer modules**: ViT Block, Deformable Attention, Transformer Block, MHSA.
- **4 Neck enhancement modules**: BiFPN, ASFF, SDLI, Gather-Distribute (Gold-YOLO).
- **3 SPP improvement modules**: SPPCSPC, SPPFCSPC, ASPP.
- **1 Head improvement module**: DyHead.
- **Tkinter desktop GUI** with 7 tab pages: Detection, Training, Validation, Export, Benchmark, Plugin Browser, Result Browser.
- **CLI scripts**: train, detect, val, export, benchmark, create_dataset.
- **Utility modules**: config parser, data augmentations, evaluation metrics, visualization, callbacks, dataset tools, logger.
- **Tutorial**: Jupyter notebook demo and plugin usage guide.
- **Configuration system**: YAML-based with command-line overrides.
- **Light/Dark theme support** in GUI.
- **Keyboard shortcuts** for common GUI operations.
- **Task queue** for sequential multi-task execution in GUI.
- **Configuration persistence** for GUI user preferences.
- **Model export** support: ONNX, TensorRT, TFLite, OpenVINO, CoreML, NCNN.

[1.0.0]: https://github.com/username/yolov8-plugins/releases/tag/v1.0.0
