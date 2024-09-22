基于 https://github.com/Aaron-CHM/ComfyUI-z-a1111-sd-webui-DanTagGen 版本修改bug，运行在铁锅炖启动器comfyui的DanTagGen

源代码(webui)来自 https://github.com/KohakuBlueleaf/z-a1111-sd-webui-dtg

以下为安装方法：

1.下载该仓库，解压到当前文件夹

2.复制 “ComfyUI-DanTagGen-main” 文件夹至 "你的整合包路径\ComfyUI_Max\ComfyUI\custom_nodes" 中(或ComfyUI_Pro、ComfyUI_Mini，与AIGODLIKE-ComfyUI-Translation在同一个文件夹中)

3.在 "你的整合包路径\ComfyUI_Max\python_embeded\Lib\site-packages“ 中查找是否有“kgen”"llama_cpp"等一系列文件夹，如果有则跳转到

4.安装kgen，在铁锅炖启动器-comfyui-配置-环境-安装依赖-输入依赖包名称与版本 中，输入 `tipo-kgen`,点击安装即可(控制台会提示“内存资源不足，无法处理此命令。”，对安装无影响，以实际提示为准)

5.尝试安装llama-cpp-python，在铁锅炖启动器-comfyui-配置-环境-安装依赖-输入依赖包名称与版本 中，输入 `llama-cpp-python`,点击安装,如出现
```bash
 CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
      CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
      -- Configuring incomplete, errors occurred!

      *** CMake configuration failed
```
等有关Cmake的错误，你需要根据以下步骤自行编译安装

6.安装并运行Microsoft C++ 生成工具 https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/ ，点选“使用C++的桌面开发”并下载安装(有6GB大小)

7.安装Cmake https://cmake.org/ ，点击右上DOWNLOAD,选择 Windows x64 Installer 对应的安装包下载并安装
  https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3-windows-x86_64.msi

8.安装scikit-build-core，在铁锅炖启动器-comfyui-配置-环境-安装依赖-输入依赖包名称与版本 中，输入 `scikit-build-core`,点击安装

9.根据步骤5再次安装llama-cpp-python，编译会需要些时间，可以查看任务管理器了解CPU运行情况，有报错可以发issue或B站私信

10.下载模型，并放在 "你的整合包路径\ComfyUI_Max\ComfyUI\custom_nodes\ComfyUI-DanTagGen-main\models" 中
  推荐DanTagGen-delta-rev2的模型，3选1：
  https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2/resolve/main/ggml-model-Q6_K.gguf?download=true （323MB）
  https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2/resolve/main/ggml-model-Q8_0.gguf?download=true （418MB）
  https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2/resolve/main/ggml-model-f16.gguf?download=true （786MB）

11.打开comfyui使用吧(｡･∀･)ﾉﾞ！