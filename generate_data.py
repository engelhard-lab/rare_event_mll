import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 2023
STEP_SIZE = 1e-3
PRINT_OUTPUT = False
PLOT = False

def generate_data_shared_features(
		n_patients, n_features, n_relevant,
		event_rate1, event_rate2,
		n_random_features,
		similarity, shared_second_layer_weights,
		random_seed=RANDOM_SEED,
		step_size=STEP_SIZE, print_output=PRINT_OUTPUT, plot=PLOT
):

	# generate N_FEATURES-dimensional feature vector for N_PATIENTS
	np.random.seed(random_seed)
	rs = np.random.RandomState(random_seed)
	x = np.random.normal(0, scale = 10, size=(n_patients, n_features))
	
	n_overlapping = round(n_random_features*similarity)
	n_distinct = round(n_random_features*(1-similarity))

	# generate coefficient matrix defining random features
	weight = random_orthogonal_set(n_random_features*2, n_relevant, rs)
	weight = weight * (np.sqrt(6/(n_random_features*2 + n_relevant)))
	W1 = np.concatenate(
		(weight[:n_random_features,:].T,
		np.zeros(shape=(n_features-n_relevant, n_random_features))),
		axis=0)

	h1 = relu(x @ W1)

	# generate coefficient vector for second layer
	c = glorot_uniform(rs, 2*n_random_features, 1)
	c1 = c[:n_random_features]

	# find logit offset that gives the desired event rate
	offset1 = find_offset(
		rs,
		np.dot(h1, c1),
		event_rate1,
		step_size
	)

	# calculate logits for each event
	l1 = np.dot(h1, c1) - offset1

	# calculate probability of each event
	p1 = sigmoid(l1)

	# generate events
	e1 = bernoulli_draw(rs, p1)

	# generate labels for event 2
	W2 = np.concatenate(
		(weight[n_random_features:,:].T,
		np.zeros(shape=(n_features-n_relevant, n_random_features))),
		axis=0)

	h2 = np.concatenate([relu(x @ W1)[:, :n_overlapping], 
					  relu(x @ W2)[:, n_overlapping:]],
					  axis=1)
	if shared_second_layer_weights:
		c2 = np.concatenate([c[:n_overlapping], c[n_random_features:n_random_features+n_distinct]])
	else:
		c2 = c1

	offset2 = find_offset(
		rs,
		np.dot(h2, c2),
		event_rate2,
		step_size
	)
	l2 = np.dot(h2, c2) - offset2
	p2 = sigmoid(l2)
	e2 = bernoulli_draw(rs, p2)

	if print_output:
		print('Dot product of u1 and u2: %.2f' % np.dot(normalize(c1),
														normalize(c2)))
		print('Avg prob')
		print(np.mean(
			[p1[i] for i in range(len(p1)) if e1[i] == 1]) / np.mean(p1))
		print(np.mean(
			[p1[i] for i in range(len(p1)) if e2[i] == 1]) / np.mean(p1))
		print('Rate of event 1: %.3f' % np.mean(e1))
		print('Rate of event 2: %.3f' % np.mean(e2))
		print('Rate of co-occurrence: %.3f' % np.mean(e1 & e2))
		print('Correlation between events: %.3f' % np.corrcoef(e1, e2)[0, 1])
		print('STD p1: %.3f' % np.std(p1))

	if plot:
		plot_logits_and_probs(l1, l2, p1, p2)

	return x, p1, p2, e1, e2

def random_orthogonal_set(n_vectors, dim, rs):
	# rank check
	if n_vectors>dim:
		return "n_vectors > dim, can't create orthogonal basis"
	else:
		vector_set = [normalize(rs.rand(dim))]

		for i in range(1, n_vectors):
			new_vector = rs.rand(dim)
			for j in range(i):
				projection = np.dot(new_vector, vector_set[j])*vector_set[j]
				new_vector -= projection
			vector_set.append(normalize(new_vector))

		return(np.array(vector_set))

def generate_data_linear(
		n_patients, n_features, n_relevant,
		event_rate1, event_rate2,
		similarity,
		random_seed=RANDOM_SEED, step_size=STEP_SIZE,
		print_output=PRINT_OUTPUT, plot=PLOT
):

	np.random.seed(random_seed)
	rs = np.random.RandomState(random_seed)
	x = np.random.normal(0, scale = 0.1, size=(n_patients, n_features))

	# generate coefficient vectors for events 1 and 2
	u1, u2 = generate_vectors_by_similarity(rs, n_features, similarity)

	# find logit offset that gives the desired event rate
	offset1 = find_offset(
		rs,
		np.dot(x[:,:n_relevant], normed_uniform(rs, n_features)[:n_relevant]),
		event_rate1,
		step_size
	)
	offset2 = find_offset(
		rs,
		np.dot(x[:,:n_relevant], normed_uniform(rs, n_features)[:n_relevant]),
		event_rate2,
		step_size
	)

	# calculate logits for each event
	l1 = np.dot(x[:,:n_relevant], u1[:n_relevant]) - offset1
	l2 = np.dot(x[:,:n_relevant], u2[:n_relevant]) - offset2

	# calculate probability of each event
	p1 = sigmoid(l1)
	p2 = sigmoid(l2)

	# generate events
	rs = np.random.RandomState(random_seed)
	e1 = bernoulli_draw(rs, p1)
	e2 = bernoulli_draw(rs, p2)

	if print_output:
		print('Dot product of u1 and u2: %.2f' % np.dot(u1, u2))
		print('Avg prob')
		print(np.mean(
			[p1[i] for i in range(len(p1)) if e1[i] == 1]) / np.mean(p1))
		print(np.mean(
			[p1[i] for i in range(len(p1)) if e2[i] == 1]) / np.mean(p1))
		print('Rate of event 1: %.3f' % np.mean(e1))
		print('Rate of event 2: %.3f' % np.mean(e2))
		print('Rate of co-occurrence: %.3f' % np.mean(e1 & e2))
		print('Correlation between events: %.3f' % np.corrcoef(e1, e2)[0, 1])

	if plot:
		plot_logits_and_probs(l1, l2, p1, p2)

	return x, p1, p2, e1, e2


def plot_logits_and_probs(l1, l2, p1, p2):
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
	u1_ = normalize(u1_ - u1 * np.dot(u1, u1_)/100)

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
	return normalize(rs.normal(loc=0, scale=100, size=n))
	# return normalize(rs.rand(n) - .5)


def bernoulli_draw(rs, p):
	return (rs.rand(len(p)) < p).astype(int)


def glorot_uniform(rs, num_in, num_out):
	scale_factor = 2 * np.sqrt(6 / (num_in + num_out))
	return scale_factor * np.squeeze(rs.rand(num_in, num_out) - .5)
	# return np.random.normal(0, 1, size=(num_in, num_out))


def logit(p):
	return np.log(p / (1 - p))


def sigmoid(l):
	return 1 / (1 + np.exp(-1 * l))


def normalize(v):
	return (v / np.linalg.norm(v))*10


def relu(v):
	return np.maximum(v, 0)

