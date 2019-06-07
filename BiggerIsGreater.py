
w = 'abcd'

def biggerIsGreater(w):
	t = 1
	while (t < len(w) and w[-t] <= w[-t-1]):
		t = t+1

	if t == len(w):
		w_new = 'no answer'
	else:
		w_new = w[:-t - 1] + min_up(w[-t-1:])

	return w_new


def min_up(wword):
	word = wword
	t = 1
	while word[0] >= word[-t]:
		t = t+1

	lst = list(word)
	lst[0], lst[-t] = lst[-t], lst[0]
	lst[1:] = lst[-1:0:-1]

	return ''.join(lst)

print(biggerIsGreater('abdc'))
print(biggerIsGreater('fedcbabcd'))
