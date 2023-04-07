---
title: setup
copyright: true
permalink: 1
top: 0
mathjax: true
date: 2021-06-24 20:25:22
tags:
categories:
password:
---
# Setup a New Machine!

Download, compile and install commonly used software to a custom path.

## Environment
- macOS Catalina 10.15.6 (19G73)
- MacBook Pro (13-inch, 2018, Four Thunderbolt 3 Ports)
- Processor 2.3 GHz Quad-Core Intel Core i5
- Memory 16 GB 2133 MHz LPDDR3
- Graphics Intel Iris Plus Graphics 655 1536 MB
```bash
Darwin 19.6.0 Darwin Kernel Version 19.6.0: Sun Jul  5 00:43:10 PDT 2020; root:xnu-6153.141.1~9/RELEASE_X86_64 x86_64
```

## Term of Usage

## Usage
```bash
# configure setup path, you can put them into your `.bashrc` or `.zshrc`
# e.g. install git

# Install oh my zsh, Oh My Zsh is an open source, community-driven framework for managing your zsh configuration.
sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# Install Vundle, Vundle is short for Vim bundle and is a Vim plugin manager.
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
# Install powerlevel
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k
# Update config.
git clone https://github.com/YinminZhang/setup.git
cp vimrc ~/.vimrc && cp zshrc ~/.zshrc
# Install plugin
vim +PluginInstall +qall
# source zsh
exec zsh
```
## Table of Contents
1. [iTerm2](#iTerm2)
2. [zsh](#zsh)
3. [vim](#vim)
4. [tmux](#tmux)

### iTerm2
#### Font
After font config, the iTerm2 can display many icons.

First, we download fonts package.
```bash
brew tap homebrew/cask-fonts
brew cask install font-hack-nerd-font
```
Then, we config the font in iTerm2.

Preferences -> Profiles -> Open Profiles -> Edit Profiles -> Text
<img src="img/iterm2/font.png"></img>

#### Transparancy & Background
We can set image as iTerm2 background.

Preferences -> Profiles -> Open Profiles -> Edit Profiles -> Window
<img src="img/iterm2/background.png"></img>

#### Advanced feature
Status bar

Preferences -> Profiles -> Open Profiles -> Edit Profiles -> Session
<img src='img/iterm2/bar.png'></img>


### zsh
Oh My Zsh is an open source, community-driven framework for managing your zsh configuration.
We can install oh my zsh trough ``curl`` or ``wget``.
```bash
# curl
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# wget
sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

I recommend some plugin to improve your develop efficiency.
```bash
# complete
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# highlight
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# autojump
git clone git://github.com/wting/autojump.git && cd autojump && ./install.py
# gitstatus
git clone --depth=1 https://github.com/romkatv/gitstatus.git ~/gitstatus
echo 'source ~/gitstatus/gitstatus.prompt.zsh' >>! ~/.zshrc
```
You should config ``plugins`` in ``~/.zshrc`` and run `source ~/.zshrc` to make sure the plugins to take effect.

**Note**: zsh-autosuggestions maybe break down, because of ``TERM type``,       ``HIGHLIGHT_STYLE``, and ``plugin confilct``. You can try some following commands:
```bash
# TERM type is not xterm-256color, you can check TERM type through command  ``env | grep TERM``
export TERM=xterm-256color
echo "export TERM=xterm-256color" >> ~/.zshrc && source ~/.zshrc
# change HIGHLIGHT_STYLE
echo "ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=yellow'" >> ~/.zshrc && source ~/.zshrc
# plugin conflict
exec zsh # instead of source ~/.zshrc
```

Theme
```bash
# powerlevel10k
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
# powerlevel9k for zsh < 5.1
git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k
```
Then, we config ``~/.zshrc``, set ``ZSH_THEME`` to ``powerlevel10k/powerlevel10k`` and source `~/.zshrc`.
### vim

```bash
# ultimate vimrc Awesome version
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh

# Vundle is short for Vim bundle and is a Vim plugin manager.
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
vim +PluginInstall +qall

# undo && redo(ctrl+r)
# echo "set undofile" >> ~/.vimrc
# echo "set undodir=~/.vim/undodir" >> ~/.vimrc
mkdir ~/.vim/undodir
```
### tmux

TBD

### plugin
[fzf](https://github.com/junegunn/fzf)