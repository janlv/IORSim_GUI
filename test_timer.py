import re
from threading import Thread
from time import sleep

# #--------------------------------------------------------------------------------
# def timer_func(func):
# #--------------------------------------------------------------------------------
#     def inner(*args, **kwargs):
#         timer = args[0]
#         while True:
#             sleep(1)
#             timer.sec += 1
#             if timer.sec == timer.limit:                
#                 func(*args, **kwargs)
#     return inner

class timer_thread:
    def __init__(self, limit=0, func=None):
        self.sec = 0
        self.func = func
        self.limit = limit
        self.running = False
        self.thread = Thread(target=self._timer, daemon=True)

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()
        else:
            self.sec = 0

    def stop(self):
        self.running = False
        self.thread.join()

    def completed(self):
        if self.sec >= self.limit:
            return True
        return False

    def _timer(self):
        while self.running:
            sleep(1)
            self.sec += 1
            if self.sec == self.limit:
                self.func()


timer = timer_thread(limit=5, func=lambda : print('HEI'))
timer.start()
for i in range(5):
    sleep()
    if timer.completed():
        print('Completed')
    else:
        print('NOT completed')
timer.stop()
#print(timer.limit_reached())
#timer.stop()
