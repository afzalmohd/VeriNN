import sys
import os
def remove_empty_lines(filename):
	with open(filename) as infile, open('modified_cons.txt', 'w') as outfile:
		for line in infile:
			if not line.strip(): continue
			outfile.write(line)  
orig_stdout = sys.stdout
f = open("original_const.txt", "r+")
s = ""
for x in f:
	if x.find(';') == -1:
		word = x.split(',')
		s_split = word[0]
		s1_split = ""
		try:
			newx = word[2].strip("\n' '")
			if newx[0] == '-':
				s1_split = newx
			else:
				s1_split = "+"
				s1_split += newx

		except:
			pass
		s += s_split
		s += s1_split
	else:
		word = x.split(';')
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
