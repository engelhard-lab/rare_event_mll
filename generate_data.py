import numpy as np

N_PATIENTS = 10000
N_FEATURES = 10
RANDOM_SEED = 2023
STEP_SIZE = 1e-2

EVENT_RATE = 0.01
SIMILARITY = 0.95


def main():

	x, e1, e2 = generate_data_linear(plot=True)

	print('Rate of event 1: %.3f' % np.mean(e1))
	print('Rate of event 2: %.3f' % np.mean(e2))
	print('Rate of co-occurrence: %.3f' % np.mean(e1 & e2))
	print('Correlation between events: %.3f' % np.corrcoef(e1, e2)[0, 1])

	x, e1, e2 = generate_data_shared_features(2, 38, shared_second_layer_weights=True, plot=True)

	print('Rate of event 1: %.3f' % np.mean(e1))
	print('Rate of event 2: %.3f' % np.mean(e2))
	print('Rate of co-occurrence: %.3f' % np.mean(e1 & e2))
	print('Correlation between events: %.3f' % np.corrcoef(e1, e2)[0, 1])


def generate_data_shared_features(
	n_distinct, n_overlapping,
	n_patients=N_PATIENTS, n_features=N_FEATURES,
	random_seed=RANDOM_SEED, event_rate=EVENT_RATE,
	step_size=STEP_SIZE, shared_second_layer_weights=True,
	plot=False):

	rs = np.random.RandomState(random_seed)

	# generate N_FEATURES-dimensional feature vector for N_PATIENTS
	x = rs.randn(n_patients, n_features)

	n_random_features = n_overlapping + 2 * n_distinct

	# generate coefficient matrix defining random features
	#W = rs.randn(n_features, n_random_features)
	W = glorot_uniform(rs, n_features, n_random_features)

	h = relu(x @ W)

	# generate coefficient vector for second layer
	c1 = glorot_uniform(rs, n_random_features, 1)

	if shared_second_layer_weights:
		c2 = c1.copy() # copying here is critical otherwise it's the same object
	else:
		c2 = glorot_uniform(rs, n_random_features, 1)

	# zero coefficients such that outcomes 1 and 2 depend on 
	# a) n_overlapping overlapping features; and
	# b) n_distinct distinct features
	c1[:n_distinct] = 0
	c2[n_distinct: (2 * n_distinct)] = 0

	# print similarity between (normed) c1 and c2
	print('Dot product of u1 and u2: %.2f' % np.dot(normalize(c1), normalize(c2)))

	# find logit offset that gives the desired event rate
	offset1 = find_offset(
		rs,
		np.dot(h, c1),
		event_rate,
		step_size
	)

	offset2 = find_offset(
		rs,
		np.dot(h, c2),
		event_rate,
		step_size
	)

	# calculate logits for each event
	l1 = np.dot(h, c1) - offset1
	l2 = np.dot(h, c2) - offset2

	# calculate probability of each event
	p1 = sigmoid(l1)
	p2 = sigmoid(l2)

	# generate events
	e1 = bernoulli_draw(rs, p1)
	e2 = bernoulli_draw(rs, p2)

	if plot:

		plot_logits_and_probs(l1, l2, p1, p2)

	return x, e1, e2


def generate_data_linear(
	n_patients=N_PATIENTS, n_features=N_FEATURES,
	random_seed=RANDOM_SEED, event_rate=EVENT_RATE,
	step_size=STEP_SIZE, similarity=SIMILARITY,
	plot=False):

	rs = np.random.RandomState(random_seed)

	# generate n_features-dimensional feature vector for N_PATIENTS
	x = rs.randn(n_patients, n_features)

	# generate coefficient vectors for events 1 and 2
	u1, u2 = generate_vectors_by_similarity(rs, n_features, similarity)

	# find logit offset that gives the desired event rate
	offset = find_offset(
		rs,
		np.dot(x, normed_uniform(rs, n_features)),
		event_rate,
		step_size
	)

	# print similarity between u1 and u2
	print('Dot product of u1 and u2: %.2f' % np.dot(u1, u2))

	# calculate logits for each event
	l1 = np.dot(x, u1) - offset
	l2 = np.dot(x, u2) - offset

	# calculate probability of each event
	p1 = sigmoid(l1)
	p2 = sigmoid(l2)

	# generate events
	e1 = bernoulli_draw(rs, p1)
	e2 = bernoulli_draw(rs, p2)

	if plot:

		plot_logits_and_probs(l1, l2, p1, p2)

	return x, e1, e2


def plot_logits_and_probs(l1, l2, p1, p2):

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))

	ax[0].hist(l1, alpha=.5, bins=20, label='Event 1')
	ax[0].hist(l2, alpha=.5, bins=20, label='Event 2')
	ax[0].set_title('Event Logits')
	ax[0].legend()

	ax[1].hist(p1, alpha=.5, bins=20, label='Event 1')
	ax[1].hist(p2, alpha=.5, bins=20, label='Event 2')
	ax[1].set_title('Event Probabilities')
	ax[1].legend()

	plt.show()


def generate_vectors_by_similarity(rs, n, s):

	# generate vector 1
	u1 = normed_uniform(rs, n)

	# generate a vector orthogonal to v1
	u1_ = normed_uniform(rs, n)
	u1_ = normalize(u1_ - u1 * np.dot(u1, u1_))

	# generate vector 2
	u2 = u1 * s + u1_ * (1 - s)

	return u1, u2


def find_offset(rs, logits, event_rate, step_size):

	offset = 0.
	rate = 1.

	while rate > event_rate:

		offset += step_size
		p = sigmoid(logits - offset)
		rate = np.mean(bernoulli_draw(rs, p))

	return offset


def normed_uniform(rs, n):
	return normalize(rs.rand(n) - .5)


def bernoulli_draw(rs, p):
	return (rs.rand(len(p)) < p).astype(int)


def glorot_uniform(rs, num_in, num_out):
	scale_factor = 2 * np.sqrt(6 / (num_in + num_out))
	return scale_factor * np.squeeze(rs.rand(num_in, num_out) - .5)


def logit(p):
	return np.log(p / (1 - p))


def sigmoid(l):
	return 1 / (1 + np.exp(-1 * l))


def normalize(v):
	return v / np.linalg.norm(v)


def relu(v):
	return np.maximum(v, 0)


if __name__ == '__main__':
	main()
