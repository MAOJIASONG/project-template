import colorlog
import logging
import sys


def build_logger(logger_name, log_filename, log_level):
    logger = colorlog.getLogger(logger_name)
    logging.root.setLevel(log_level)

    # define console handler
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
            }
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_formatter = colorlog.ColoredFormatter(fmt_string, log_colors=log_colors)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # define file handler
    file_handler = logging.FileHandler(log_filename, mode='a')  # 追加模式写入日志文件
    file_fmt_string = '%(asctime)s - %(levelname)s - %(message)s'  # 去掉颜色信息
    file_formatter = logging.Formatter(file_fmt_string)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


    # configure other libraries
    import datasets
    import transformers
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger