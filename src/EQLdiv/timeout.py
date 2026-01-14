import signal
import threading
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    if hasattr(signal, "SIGALRM"):
        def signal_handler(signum, frame):
            raise TimeoutException("Timeout after {} seconds.".format(seconds))

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        timer = threading.Timer(seconds, threading.interrupt_main)
        timer.daemon = True
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutException("Timeout after {} seconds.".format(seconds))
        finally:
            timer.cancel()
