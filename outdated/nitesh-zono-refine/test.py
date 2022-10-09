st = set()
i = 10
def check():
	global i
	i = 11
	print(i)
if __name__ == "__main__":
    check()
    print(i)