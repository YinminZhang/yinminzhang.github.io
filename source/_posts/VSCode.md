---
title: VSCode
copyright: true
permalink: 1
top: 0
mathjax: true
date: 2021-01-10 22:27:46
tags: diary
categories: diary
password:
---
# vscode configure
## Extension
### Python
- Python 
    Python extension for Visual Studio Code.
- Python Indent
    Correct python indentation in Visual Studio Code. 
- Python snippets
    Python snippets collections.
- Python Docstring Generator
    Visual Studio Code extension to quickly generate docstrings for python functions.
### Remote
- SFTP
    sftp sync extension for VS Code.
- Remote - SSH
    The Remote - SSH extension lets you use any remote machine with a SSH server as your development environment.

### C++
- C/C++
    The C/C++ extension adds language support for C/C++ to Visual Studio Code, including features such as IntelliSense and debugging.
- CodeLLDB
    Debugging on Linux (x64 or ARM), macOS and Windows.
- C/C++ Clang Command Adapter
    Completion and Diagnostic for C/C++/Objective-C using Clang command.
- clangd
    Provides C/C++ language IDE features for VS Code using clangd:
### Autocomplete and IntelliSense
- Visual Studio IntelliCode
- Kite Autocomplete Plugin for Visual Studio Code

### Others
- Jupyter Extension for Visual Studio Code
- IntelliJ IDEA Key Bindings
    Port of IntelliJ IDEA key bindings for VS Code. Includes keymaps for popular JetBrains products like IntelliJ Ultimate, WebStorm, PyCharm, PHP Storm, etc.
- language-stylus
    Adds syntax highlighting and code completion to Stylus files in Visual Studio Code.
- Markdown PDF
    This extension converts Markdown files to pdf, html, png or jpeg files.
- Markdown Preview Enhanced
    Markdown Preview Enhanced is an extension that provides you with many useful functionalities such as automatic scroll sync, math typesetting, mermaid, PlantUML, pandoc, PDF export, code chunk, presentation writer, etc. 
- LaTeX Workshop
    LaTeX Workshop is an extension for Visual Studio Code, aiming to provide core features for LaTeX typesetting with Visual Studio Code.

## Configure
### Python
#### Configure environment
Select environment for python.
```command + p -> >select interpreter``` 
or
```command + shift + p -> select interpreter```

#### Debug
1. Extension
- Python
2. configure ```launch.json``` and save ```launch.json```.
```json
{
    // Python debug configurations in Visual Studio Code: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "python: debug", // Provides the name for the debug configuration that appears in the VS Code drop-down list.
            "type": "python",
            "request": "launch", // ['launch', 'attach'] 
            //  launch: start the debugger on the file specified in program
            //  attach: attach the debugger to an already running process. 
            "cwd": "${workspaceFolder}", // Specifies the current working directory for the debugger, which is the base folder for any relative paths used in code.
            "program": "${file}", // Provides the fully qualified path to the python program's entry module (startup file). 
            "console": "integratedTerminal", // Specifies how program output is displayed 
            "args": [
                "--config",
                "configcenternet3d_2x.yaml"
            ] // Specifies arguments to pass to the Python program.
        }
    ]
}
```
recommand configure for local debugging as following:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "python: debug",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}",
            "program": "${workspaceFolder}/relative/path/filename.py ",
            "console": "integratedTerminal",
            "args": [
                "--config",
                "configcenternet3d_2x.yaml"
            ]
        }
    ]
}
```
3. ```command + 5``` switch to ```RUN``` or ```f5```.


### C++
#### Configure PATH
1. search include path list.
```shell
gcc -v -E -x c -  
```
2. ```command + shift + p -> c/c++" Edit Configure```
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/00055f34-7eec-41dd-913c-2f909ca618df/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210110%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210110T131302Z&X-Amz-Expires=86400&X-Amz-Signature=526d9048ed969e130f26bbf304ae7b0f1798620fffd334c61a03e40f989e6249&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>

3. type ```include path list``` to ```include path```.
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/adc0db32-2834-4095-a0e0-9c507b346c2b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210110%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210110T131507Z&X-Amz-Expires=86400&X-Amz-Signature=414c63fe4af917b88282b97f62e02151741c12d7bec6373c6efa91c6debb0f10&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>

#### Debug
1. Extension
- C/C++
- C/C++ Clang Command Adapter
- CodeLLDB
2. configure ```launch.json``` and ```tasks.json```.
launch.json
```json
{
    "name": "Launch",
    "type": "lldb",
    "request": "launch",
    "program": "${workspaceFolder}/{fileBasenameNoExtension}", // ${workspaceFolder}/<my program>
    "args": ["-arg1", "-arg2"],
    "preLaunchTask": "Build with Clang"
}
```
tasks.json
```json
{
    "tasks": [
        {
            "label": "Build with Clang", // same with launch["preLaunchTask"]
            "type": "shell",
            "command": "clang++",
            "args": [
                "-std=c++17",
                "-stdlib=libc++",
                "${fileBasenameNoExtension}.cpp",
                "-o",
                "${fileBasenameNoExtension}",
                "--debug"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ],
    "version": "2.0.0"
}
```
3. ```command + 5``` switch to ```RUN``` or ```f5```.
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fcb01dda-7ea6-44d1-bd91-f4b88b5ad5f6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210110%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210110T130153Z&X-Amz-Expires=86400&X-Amz-Signature=624392db9c273f4557c8131eae1e442c335ba33fb122fdff93867567e983dc34&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>

### Remote
#### Remote - SSH
- Access remote server to modify, upload and download files.
- Access remote server to run file(python, cpp etc.).

1. ```command + shift + p -> remote-ssh: open configuration file```
select config path.
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9fbfc512-05a8-4be9-bfaf-68b8656c18e3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210110%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210110T133102Z&X-Amz-Expires=86400&X-Amz-Signature=1d6c0586a4d675f4fd8697cf5c29dc32a4fc8633e8a4e41dfcfe2b3506b6830a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>

2. config
```shell
# Read more about SSH config files: https://linux.die.net/man/5/ssh_config
Host Name
    HostName ip # Specifies the real host name to log into. For example, 220.181.38.150
    User zhangsan # username Specifies the user to log in as.
    Port 22 # Specifies the port number to connect on the remote host. The default is 22.
```

3. select server and file directory to access.
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1b68f76a-8e8b-4a1b-bce2-1c0ec57d185f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210110%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210110T133913Z&X-Amz-Expires=86400&X-Amz-Signature=5613bdb26d8ca8f1e9c418c81b9c70eaeccb33b89dca29fda21ff379f9604195&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>