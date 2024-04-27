---
title: "Poetry `No such file or directory: 'python'` Error on MacOS"
collection: portfolio
---

Recently, I had to do some work with LangChain and wanted to create an environment
with the `poetry install` command. When I ran the command, I got the following
error:

```
$ poetry install

[Errno 2] No such file or directory: 'python'
```

Browsing the web, I came accross the `python-is-python3` package in several
places [[1](https://github.com/python-poetry/poetry/issues/6841#issuecomment-1401151255),
[2](https://stackoverflow.com/a/61921941)]. However, I didn't want to install
this package.

First reason was that I used `brew` to install packages and formula for
`python-is-python3` didn't exist in the brew repository. Everything you
can download in `brew` needs a formula ([see](https://itssiva.medium.com/difference-between-update-and-upgrade-on-brew-apt-get-etc-8a227fcbbd45)).
Second reason was that it felt excessive to download a whole package for
this error.

Upon more research, I came across [brew documentation on Python](https://docs.brew.sh/Homebrew-and-Python).
The documentation refers to symlinks for `python`, `pip` etc:

> Unversioned symlinks for python, python-config, pip etc. are installed here:
> ```
> $(brew --prefix python)/libexec/bin
> ```

I checked where my python was installed by running `brew --prefix python` and
replaced the directory in the command above. I then added this to my path in
`.zshrc`.

First, run the following to edit the `.zshrc` file:

```
nano ~/.zshrc
```

Add the following line to update the `PATH` (after replacing `<python-prefix>`
with the result of `brew --prefix python`):

```
export PATH="$PATH:<python-prefix>/libexec/bin"
```

This will add the symlinks to your path, and you will be able to run `pip` or
`python` in your console. This will also solve the error in poetry.
