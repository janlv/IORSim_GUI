from IORlib.utils import timer_thread
from time import sleep

with timer_thread(limit=5, prec=0.5, func=lambda : print('HEI')) as timer:
    for i in range(10):
        timer.start()
        sleep(7)
    timer.close()
