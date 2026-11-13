"""
log_utils.py —— 带「文件名:行号」前缀的打印辅助
==============================================

用 log(...) 代替 print(...)，输出会自动带上「调用处的文件名:行号」前缀，
方便对照代码查看执行情况。例如：

    [main.py:48] Connected to turtlesim (/turtle1) ...

实现方式：通过调用栈上一层的帧（frame）拿到调用者的文件名与行号，
其余参数与内置 print 完全一致（sep / end / file / flush 都透传）。

注意：动画型输出（如 cli_spinner.py 用 \\r 刷新同一行）不应使用本函数，
以免前缀破坏行内刷新效果。
"""

from __future__ import annotations

import inspect
import os
import sys


def log(*args, **kwargs) -> None:
    """等价于 print()，但在最前面加上「[文件名:行号]」前缀。

    透传 print 的全部关键字参数（sep、end、file、flush）。
    """
    # 取调用者（上一层帧）的文件名与行号
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is not None:
        filename = os.path.basename(caller.f_code.co_filename)
        lineno = caller.f_lineno
        prefix = f"[{filename}:{lineno}]"
    else:
        prefix = "[?]"

    file = kwargs.get("file", sys.stdout)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)

    # 前缀后接一个空格，再接原本要打印的内容
    text = sep.join(str(a) for a in args)
    print(f"{prefix} {text}", end=end, file=file, flush=flush)
