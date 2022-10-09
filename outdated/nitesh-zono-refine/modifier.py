import sys
import os
def remove_empty_lines(filename):
	with open(filename) as infile, open(sys.argv[2], 'w') as outfile:
		for line in infile:
			if not line.strip(): continue
			outfile.write(line)  
orig_stdout = sys.stdout

#f = open("testing.txt", "r+")


'''

It will take file name from argv[1] and modify
constrainsts to required format and save these
constraints to file given in argv[2]

'''

f = open(sys.argv[1], 'r+')
s = ""
for line in f:
	if line.find(';') == -1:
		word = line.split(',')
		s_split = word[0]
		s1_split = ""
		try:
			newx = word[2].strip("\n' '")
			'''if newx.find('-') == -1:
				s1_split = newx
			else:'''
			s1_split = "+"
			s1_split += newx

		except:
			pass
		s += s_split
		s += s1_split
	else:
		word = line.split(';')
		s += word[0].strip()
		s += '\n'
		s += word[1].split(']')[0] + ']'
		s += '\n'
t = s.replace('[', '')
u = t.replace(']', '')
f = open('out.txt', 'r+')
sys.stdout = f
print(u)
sys.stdout = orig_stdout
f.close()
remove_empty_lines('out.txt')
f = open('out.txt', 'r+')
f.truncate(0)
f.close()