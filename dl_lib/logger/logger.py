import logging
import logging.handlers
import traceback
from typing import Callable, Tuple
import multiprocessing as mp


class MultiProcessLoggerListener:
    """
    Using a independent process to write logger, suitable for multiprocess logging.
    One can use queue handler to write log from other process, e.g.
    ```
        name = multiprocessing.current_process().name
        h = logging.handlers.QueueHandler(MultiProcessLogger.queue)
        logger = logging.getLogger("worker_process.{}".format(name))
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
    ```
    """
    def __init__(self, logger_constructor: Callable[[], Tuple[logging.Logger, str]], mp_context="spawn"):
        """
        Args:
            logger_constructor: function to create root logger
        """
        mmp = mp.get_context(mp_context)
        self._queue = mmp.Queue(-1)
        self._logger_constructor = logger_constructor
        self.listener = mp.Process(target=self.listen, name="listener")
        self.listener.start()

    def listen(self):
        root, _ = self._logger_constructor()
        root.info("Starting...")
        while True:
            try:
                record = self._queue.get()
                if record is None:
                    root.info("Stopped")
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
            except Exception as e:
                tb = traceback.format_exc()
                root.fatal("Exception:\n{}\nTraceback:\n{}".format(e, tb))

    @property
    def queue(self) -> mp.Queue:
        return self._queue

    def stop(self):
        """
        Stop listener process and join
        """
        self._queue.put_nowait(None)
        self.join()

    def join(self):
        """
        Wait for listener stop
        """
        self.listener.join()

    def get_logger(self, name: str = None, level: int = logging.INFO):
        logger = logging.getLogger(name)
        handler = logging.handlers.QueueHandler(self._queue)
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    def __del__(self):
        self.stop()
