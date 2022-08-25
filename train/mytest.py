import sys
import os
def a(x):
    print('this is a', x)
if __name__ == '__main__':
    config_dict = read_config()
    x = []
    if len(sys.argv) > 1:
        x = sys.argv[1:]
        print(x)
        time.sleep(1) # 休眠1秒
    print(config_dict)
