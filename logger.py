import logging

fmt = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)
fmt = logging.Formatter(fmt=fmt)

logger = logging.getLogger("train")
handler = logging.FileHandler("logger.log")
handler.setFormatter(fmt)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
