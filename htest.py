from scipy.stats import multinomial
from heapq import heappush, heappop
import pdb

def min_conf_reg(p, level):
	""" 
	Return the smallest confidence region for p and the given level

	p - a multinomial probability distribution
	level - the likelihood of that a sample from p falls into the region
	"""
	h = []
	visited = {}
	total = 0.0
	totaled = []
	current = most_likely(p)
	#print(f"most likely: { current }, pmf: { p.pmf(current) }")
	pp = p.pmf(current)
	heappush(h, (-pp, current))
	visited[tuple(current)] = pp
	while total < level and len(h) > 0:
		pmf, current = heappop(h)
		pmf = -pmf
		total += pmf
		totaled.append(current)
		#print (f"tuple {current} total {total}")
		h = push_successors(h, p, current, visited)
	return totaled

def most_likely(p):
	"""
	returns the most likely multinomial sample from a given 
	scipy.stats.multinomial object, using a greedy algorithm
	that allocates irregular "chunks" of the sample per 
	iteration.

	p - the multinomial to sample from

	returns - the sample (d)
	"""
	d = [0]*len(p.p)	# the sample to return
	left = int(p.n)       	# the amount of the sample left to take
	while left > 0:
		#pdb.set_trace()
		# calculate marginal change to distribution for sampling next choice
		marginals = [(p.n-left+1) * p.p[i]/(d[i]+1) for i in range(len(p.p))] 
		maxes = sorted(marginals, reverse=True)
		maxd = marginals.index(maxes[0])
		incrs = [int((d[i] + 1) * p.p[maxd] / p.p[i] - d[maxd]) if (i != maxd and p.p[i] > 0) else left for i in range(len(p.p))]
		incr = max(1,min(incrs))
		#print (incr)
		d[maxd] += incr
		left -= incr
	return d
	

def more_likely_than(p, q):
	"""
	Returns all samples of p that are more likely than q
	"""
	h = []
	visited = {}
	total = 0.0
	totaled = {}
	current = most_likely(p)
	print(f"most likely: { current }, pmf: { p.pmf(current) }")
	pp = p.pmf(current)
	if pp > p.pmf(q):
		heappush(h, (-pp, current))
		visited[tuple(current)] = pp
	while len(h) > 0:
		pmf, current = heappop(h)
		pmf = -pmf
		total += pmf
		totaled.put(current)
		print (f"tuple {current} total {total}")
		h = push_successors(h, p, current, visited,q)
	return total

def push_successors(h, p, current, visited, q = None):
	"""
	Push a list of unexplored samples onto priority queue
	"""
	for i in [i for i in range(len(current)) if current[i] > 0]: 
		for j in [j for j in range(len(current)) if j != i]:
			newc = current.copy()
			newc[i] += -1
			newc[j] += 1
			pmf = p.pmf(newc)
			if tuple(newc) not in visited and (q == None or pmf > p.pmf(q)) and pmf > 0:
				heappush(h, (-pmf, newc))
				visited[tuple(newc)] = pmf

	return h

