// GRU with Minimal Gated Units
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
//#include<err.h>
#include <time.h>
#include <unistd.h>

#define SUCC 1
#define FAIL -1

#define MAXVOCS 100	// max vocabulary size

int ae_load_file_to_memory(const char *fname, char **result) { 
	int size = 0;
	FILE *f = fopen(fname, "rb");
	if (f == NULL) { 
		*result = NULL;
		return -1; // -1 means file opening fail 
	} 
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) { 
		free(*result);
		return -2; // -2 means file reading fail 
	} 
	fclose(f);
	(*result)[size] = 0;
	return size;
}

int file_exists(const char *fname) {
	if( access(fname, F_OK) != -1 ) {
		return SUCC;
	} else {
		return FAIL;
	}
}

int write_bin_arr(const char *fname, int rows, int cols, double *data) {
   	FILE *fptr;
	int written = 0;
   	fptr = fopen(fname,"wb");
   	if(fptr == NULL) {   
      	return FAIL;             
   	}
	written = fwrite(data, sizeof(double), rows*cols, fptr);
	if(written == 0) {
		fclose(fptr);
		return FAIL;
	}
   	fclose(fptr);
   	return SUCC;
}

int read_bin_file_arr(const char *fname, int rows, int cols, double *arr) { 
	FILE *fptr = fopen(fname, "rb");
	if (fptr == NULL) { 
		return FAIL; // file opening fail 
	}
	int rres = fread(arr, sizeof(double), rows*cols, fptr);
	return SUCC;
}

// file I/O for weights
int write_weights(int nn, int h_size, int o_size, double *Wz, double *Uz, double *Wr, // write all weights to file
				double *Ur, double *Wh, double *Uh, double *Wy) {
	int fres;
	fres = write_bin_arr("Wz.data", h_size, nn, Wz);
	if(fres<0)
		return FAIL;
	//printf("Wz: \n");
	//print(Wz, h_size, nn);
	fres = write_bin_arr("Uz.data", h_size, h_size, Uz);
	if(fres<0)
		return FAIL;
	//printf("Uz: \n");
	//print(Uz, h_size, nn);
	fres = write_bin_arr("Wr.data", h_size, nn, Wr);
	if(fres<0)
		return FAIL;
	//printf("Wr: \n");
	//print(Wr, h_size, nn);
	fres = write_bin_arr("Ur.data", h_size, h_size, Ur);
	if(fres<0)
		return FAIL;
	//printf("Ur: \n");
	//print(Ur, h_size, nn);
	fres = write_bin_arr("Wh.data", h_size, nn, Wh);
	if(fres<0)
		return FAIL;
	//printf("Wh: \n");
	//print(Wh, h_size, nn);
	fres = write_bin_arr("Uh.data", h_size, h_size, Uh);
	if(fres<0)
		return FAIL;
	//printf("Uh: \n");
	//print(Uh, h_size, nn);
	fres = write_bin_arr("Wy.data", o_size, h_size, Wy);
	if(fres<0)
		return FAIL;
	//printf("Wy: \n");
	//print(Wy, h_size, nn);
	return SUCC;
}

int read_weights(int nn, int h_size, int o_size, double *Wz, double *Uz, double *Wr, // read all weights from file
				double *Ur, double *Wh, double *Uh, double *Wy) {
	int fres;
	fres = read_bin_file_arr("Wz.data", h_size, nn, Wz);
	//printf("loading Wz: %f\n", Wz[0]);
	//printf("Wz: \n");
	//print(Wz, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Uz.data", h_size, h_size, Uz);
	//printf("Uz: \n");
	//print(Uz, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Wr.data", h_size, nn, Wr);
	//printf("Wr: \n");
	//print(Wr, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Ur.data", h_size, h_size, Ur);
	//printf("Ur: \n");
	//print(Ur, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Wh.data", h_size, nn, Wh);
	//printf("Wh: \n");
	//print(Wh, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Uh.data", h_size, h_size, Uh);
	//printf("Uh: \n");
	//print(Uh, h_size, nn);
	if(fres<0)
		return FAIL;
	fres = read_bin_file_arr("Wy.data", o_size, h_size, Wy);
	//printf("Wy: \n");
	//print(Wy, h_size, nn);
	if(fres<0)
		return FAIL;
	return SUCC;
}

int compare_char (const void *a, const void *b) {
    if (*(char *)a != *(char *)b)
        return *(char *)a - *(char *)b;

    return 0;
}

int charset(char *rawtxt, char *result) { // returns a sorted list of characters in text
	char *e;
	int len = strlen( rawtxt );
	int charsadded = 0;
	for(int i=0; i<len; i++) {
		e = (char *)strchr(result, (char)rawtxt[i]);
		if(e == NULL) {
			if(charsadded < MAXVOCS) {
				result[charsadded] = rawtxt[i];
				result[charsadded+1] = '\0';
				charsadded++;
			}
		}
	}
	qsort(result, charsadded, sizeof(char), compare_char);
	return charsadded;
}

int char_to_ix(char c, char *cset) {
	int len = strlen(cset);
	for(int i=0; i<len; i++) {
		if(cset[i] == c) {
			return i;
		}
	}
	return FAIL;
}

int ixset_to_char(int *ix, int nix, char *cset, char *res) {
	int i;
	for(i=0; i<nix; i++) {
		res[i] = cset[ix[i]];
	}
	res[i+1] = '\0';
	return SUCC;
}

int rnddchoice(int ix[], int nix, double pdist[], int npd) { // chooses a value from the array by probability distr given
	if(nix <= 0 || nix != npd)
		return FAIL;
	//srand(time(NULL));
	double sumpd = 0.;
	for(int i=0; i<npd; i++) {
		sumpd += pdist[i];
	}
	if(sumpd <= 0.)
			return FAIL;
	double myrnd = (double)rand()/RAND_MAX;
	double pdcum = 0.;
	for(int i=0; i<npd; i++) {
		pdcum += pdist[i]/sumpd;
		if(myrnd < pdcum)
			return ix[i];
	}
	return ix[nix-1];
}

int randdmat(double *m, int r, int c, double f) { // random matrix
	int i, j;
	//srand(time(NULL));
	for(i = 0; i < r; i++) {
		for(j = 0; j < c; j++) {
			m[i*c+j] = (double)rand()/RAND_MAX*f-f/2.;
		}
	}
	return SUCC; 
}

int zerodmat(double *m, int r, int c) { // zero matrix
	int i, j;
	for(i = 0; i < r; i++) {
		for(j = 0; j < c; j++) {
			m[i*c+j] = 0.;
		}
	}
	return SUCC; 
}

int print(double *m, int r, int c) { // print matrix
	int i, j;
	if(m == NULL)
		return FAIL;
	for(i = 0; i < r; i++){
		for(j = 0; j < c; j++){
			printf("%f ", m[i*c+j]);
		}
		printf("\n");
	}
	return SUCC;
}

int printv(double *m, int r, int c, int ip, int rorc) { // print vactor or matrix row/col (rorc=0: row)
	int i;
	if(m == NULL)
		return FAIL;
	if(rorc == 0)	// row
		for(i = 0; i < c; i++)
			printf("%f ", m[ip*c+i]);
	else			// col
		for(i = 0; i < r; i++)
			printf("%f ", m[i*c+ip]);
	printf("\n");
	return SUCC;
}

int dott(double *m0, int r0, int c0, int t0, double *m1, int r1, int c1, int t1, double *mres) { // dot product (and matrix multiplication) w/ transposed
	int i, j, k;
	if(m0 == NULL || m1 == NULL)
		return FAIL;
	if(c0 != r1)
		return FAIL;

	for(i = 0; i < r0; i++){
		for(j = 0; j < c1; j++){
			mres[i*c1+j] = 0.;
			for(k = 0; k < c0; k++){
				if(t0!=0){
					if(t1!=0)
						mres[i*c1+j] += m0[k*c0+i] * m1[j*c1+k]; // m0 and m1 transposed
					else
						mres[i*c1+j] += m0[k*c0+i] * m1[k*c1+j]; // m0 transposed, not m1
				} else {
					if(t1!=0)
						mres[i*c1+j] += m0[i*c0+k] * m1[j*c1+k]; // m0 not, m1 transposed
					else
						mres[i*c1+j] += m0[i*c0+k] * m1[k*c1+j]; // neither m0 nor m1 transposed
				}
			}
		}
	}
	return SUCC;
}

int addsub(int addsub, double *m0, int r0, int c0, double *m1, double *mres) { // add and subtract (plain: no row/col check)
	int i, j;
	if(m0 == NULL || m1 == NULL)
		return FAIL;

	for(i = 0; i < r0; i++){
		for(j = 0; j < c0; j++){
			if(addsub == 0)
				mres[i*c0+j] = m0[i*c0+j] + m1[i*c0+j];
			else
				mres[i*c0+j] = m0[i*c0+j] - m1[i*c0+j];
		}
	}
	return SUCC;
}

double summ(double *m0, int r0, int c0) { // sum all elements of matrix and return total
	double res = 0.;
	int i, j;
	if(m0 == NULL)
		return FAIL;

	for(i = 0; i < r0; i++){
		for(j = 0; j < c0; j++){
			res += m0[i*c0+j];
		}
	}
	return res;
}

double sigmoid(double x) {
	double addnum = 1.e-8;
	double exp_val;
	double ret_val;
	if(x >= 0.)
		exp_val = 1. / ( exp( x ) + addnum );
	else
		exp_val = exp( -x );
	ret_val = 1. / (1. + exp_val);
	return ret_val;
}

static void softmax(double *input, size_t input_len, double *output) { // size_t used to represent the size of an object
  	assert(input);
  	// assert(input_len >= 0);  Not needed
  	double m = -INFINITY;
  	for(size_t i = 0; i < input_len; i++) { // find maximum in inputs (protexts against over/underflow)
    	if (input[i] > m) {
      		m = input[i];
    	}
  	}
  	double sum = 0.0;
  	for(size_t i = 0; i < input_len; i++) { // summation and divide by exp(max) numerically stable
    	sum += exp(input[i] - m);
  	}
  	double offset = m + log(sum);
  	for(size_t i = 0; i < input_len; i++) { // multiply exp(max) back out
    	output[i] = exp(input[i] - offset);
  	}
}

// softmax: has to handle two different array types, careful with numerical stability
static int softmax2(double *input, int ir, int ic, int ithisc, double *output, int or, int oc, int othisc) {
	if(ir!=or || ithisc>ic)
		return FAIL;
	double m = -INFINITY;
	for(int i=0; i<ir; i++) { // find maximum in inputs (protects against over/underflow)
		if(input[i*ic+ithisc] > m) {
			m = input[i*ic+ithisc];
		}
	}
	double sum = 0.0;
	for(int i=0; i<ir; i++) { // summation and divide by exp(max) numerically stable
		sum += exp(input[i*ic+ithisc] - m);
	}
	double offset = m + log(sum);
	for(int i=0; i<ir; i++) { // multiply exp(max) back out
		output[i*oc+othisc] = exp(input[i*ic+ithisc] - offset);
	}
}

double mytanh(double x) { // faster tanh (or relu)
	if (x>1.0) return 1.0;
	if (x<-1.0) return -1.0;
	//double x2 = x*x;
	//double x3 = x2*x/3.;
	//return x - x3 + 2.*x3*x2/5. - 17.*x3*x3*x/35. + 62.*x3*x3*x3/105. - 1382.*x3*x3*x3*x2/5775.;
	//return x - x3 + 2.*x3*x2/5. - 17.*x3*x3*x/35.;
	//return tanh(x);
	return x;
}

static double limval(double val, double ll, double ul) { // limit val to range [ll,ul]
	if(val > ul)
		return ul;
	if(val < ll)
		return ll;
	return val;
}

static int limvalm(double *m0, int r0, int c0, double ll, double ul) { // limit matrix values to range [ll,ul]
	int i, j;
	if(m0 == NULL)
		return FAIL;

	for(i = 0; i < r0; i++){
		for(j = 0; j < c0; j++){
			if(m0[i*c0+j] > ul)
				m0[i*c0+j] = ul;
			if(m0[i*c0+j] < ll)
				m0[i*c0+j] = ll;
		}
	}
	return SUCC;
}

static int watchdog(double *m0, int r0, int c0, double threshold, char *msg, int printm) { // checks if values of matrix exceed a threshold
	int i, j;
	if(m0 == NULL)
		return FAIL;

	for(i = 0; i < r0; i++){
		for(j = 0; j < c0; j++){
			if(m0[i*c0+j] > threshold) {
				printf("watchdog: %s\n", msg);
				if(printm==1)
					print(m0, r0, c0);
				return SUCC;
			}
			if(m0[i*c0+j] < -threshold) {
				printf("watchdog: %s\n", msg);
				if(printm==1)
					print(m0, r0, c0);
				return SUCC;
			}
		}
	}
	return SUCC;
}

long randint(int lower, int upper) { // simple random integer
	double rf = (double)rand() / (double)RAND_MAX * (double)( upper - lower) + lower;
	return (long)rf;
}

// *************************************************** MAIN **************************************************************
int main (int argc, char **argv) { // use: xxxxx

	int seq_length = 25; // Longer sequence lengths allow for lengthier latent dependencies to be trained, indexed as t (corresponds to time)
	long max_nruns = 100; // number of iterations
	long print_interval = 10; // lets see....

	// init random generator (only once!)
	srand(time(NULL));
	// load text file into memory
	char *data; // corpus stored here
	int fres = ae_load_file_to_memory("input1.txt", &data);
	if( fres < 0 )
		return FAIL;
	long data_size = strlen(data);
	char *cset = (char *)malloc(MAXVOCS+1);	// create character map
	long vocab_size = charset(data, cset);
	printf("%d chars found: %s in file of length %d\n", strlen(cset), cset, data_size);
	
	long h_size = 100; // hidden layer size (100)
	double learning_rate = 0.1; // very sensitive, improvement after change from 0.1 to 0.01

	int sample_ix[vocab_size];
	for(int i=0; i<vocab_size; i++)
		sample_ix[i] = i;

	double smooth_loss = 0.;

	long pos = 0;
	long nruns = 0;

	double *Wxh;
	double *Whh, *bh;
	double *Why, *by;
	
	Wxh = malloc(h_size * vocab_size * sizeof *Wxh);
	Whh = malloc(h_size * h_size * sizeof *Whh);
	Why = malloc(vocab_size * h_size * sizeof *Why);
	bh = malloc(h_size * 1 * sizeof *bh);
	by = malloc(vocab_size * 1 * sizeof *by);

	// Model parameter initialization
	randdmat(Wxh, h_size, vocab_size, 0.01);
	randdmat(Whh, h_size, vocab_size, 0.01);
	randdmat(Why, vocab_size, h_size, 0.01);

	// init biases
	zerodmat(bh, h_size, 1);
	zerodmat(by, vocab_size, 1);

    // Initialize variables
    double xs[seq_length][vocab_size]; // xs[t][inputs[t]] 
	double hs[seq_length][h_size];
	double ys[seq_length][vocab_size];
	double ps[seq_length][vocab_size];
	double hprev[h_size];
	double dy[vocab_size];
	double dh[h_size];
	double dhraw[h_size];

    // Parameter gradient initialization
	double *dWxh;
	double *dWhh, *dbh;
	double *dWhy, *dby;
	double *dhnext;
	
	dWxh = malloc(h_size * vocab_size * sizeof *dWxh);
	dWhh = malloc(h_size * h_size * sizeof *dWhh);
	dWhy = malloc(vocab_size * h_size * sizeof *dWhy);
	dbh = malloc(h_size * 1 * sizeof *dbh);
	dby = malloc(vocab_size * 1 * sizeof *dby);

	dhnext = malloc(h_size * 1 * sizeof *dhnext);

	// memory for adagrad
	double *mWxh;
	double *mWhh, *mbh;
	double *mWhy, *mby;
	mWxh = malloc(h_size * vocab_size * sizeof *mWxh);
	mWhh = malloc(h_size * h_size * sizeof *mWhh);
	mWhy = malloc(vocab_size * h_size * sizeof *mWhy);
	mbh = malloc(h_size * 1 * sizeof *mbh);
	mby = malloc(vocab_size * 1 * sizeof *mby);
	zerodmat(mWxh, h_size, vocab_size);
	zerodmat(mWhh, h_size, h_size);
	zerodmat(mWhy, vocab_size, h_size);
	zerodmat(mbh, h_size, 1);
	zerodmat(mby, vocab_size, 1);

	// init gradients
	zerodmat(dbh, h_size, 1);
	zerodmat(dby, vocab_size, 1);
	zerodmat(dhnext, h_size, 1);

    double sequence_loss = 0.;
	smooth_loss = -log(1.0/vocab_size)*seq_length; // smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

	// while true
	//while(nruns < (data_size - seq_length)) { // step through entire text
	while (nruns < max_nruns) {
	    // Reset memory if appropriate - reset position to 0
	    if(pos + seq_length + 1 >= data_size || nruns==0 ) {
	        //hprev = np.zeros((h_size, 1)) # np.zeros(shape) -> (vocab_size, 1) = (rows, cols)
			zerodmat(hprev, h_size, 1);
	        pos = 0; // current position
		}

		// Get input and target sequence - each an index list of characters
	    //inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]] // current position to sequence length
	    //targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]] // current position+1 to sequence length+1 (next state)
		char inputs[seq_length];
		char targets[seq_length];
		for(int i=0; i<seq_length; i++) {
			inputs[i] = char_to_ix(data[pos+i], cset);		// transition from this state (=current pos) to next state (pos+1)
			targets[i] = char_to_ix(data[pos+i+1], cset);
			//printf("inputs: %d for %c\n", inputs[i], data[pos+i]);
			//printf("targets: %d for %c\n", targets[i], data[pos+i+1]);
		}

		sequence_loss = 0.;

		//printf("fwd prop \n");

	    // Forward prop ------------------------------------------------------------------------
	    for(int t=0; t<seq_length; t++) { // lossFun(inputs, targets, hprev)

			for(int i=0; i<vocab_size; i++) { // nrows
				xs[t][i] = 0.;
			}
	        xs[t][inputs[t]] = 1.; //x[t][inputs[t]] = 1; // set t'th input to one (current word in one-hot) - x[t] is a column vector

			//printv(*xs, seq_length, vocab_size, t, 0);

			for(int i=0; i<h_size; i++) {
				hs[t][i] = 0.;
			}
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<vocab_size; j++) {
					hs[t][i] += Wxh[j*vocab_size+i] * xs[t][j];		// np.dot(Wxh, xs[t]) - hs[t] is a column vector
				}
			}

			//print(Wxh, h_size, vocab_size);
			//print(*hs, seq_length, h_size);

			double temphs[h_size];
			for(int i=0; i<h_size; i++)
				temphs[i] = 0.;
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<h_size; j++) {
					if(t>0)
						temphs[i] += Whh[j*h_size+i] * hs[t-1][j];		// np.dot(Whh, hs[t-1])
					else
						temphs[i] += Whh[j*h_size+i] * hprev[j];		// np.dot(Whh, hs[t-1])
				}
			}

			for(int i=0; i<h_size; i++) {
				if(t>0)
					hs[t][i] += mytanh(temphs[i]+bh[i]); 	// hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
				else
					hprev[i] += mytanh(temphs[i]+bh[i]); 	// hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
			}

			for(int i=0; i<vocab_size; i++)
				ys[t][i] = 0.;
			for(int i=0; i<vocab_size; i++) {
				for(int j=0; j<h_size; j++) {
					ys[t][i] += Why[i*h_size+j] * hs[t][j];		// np.dot(Why, hs[t])
				}
				ys[t][i] += by[i];			// ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars - column vector
			}
			
			double esum = 0.;
			for(int i=0; i<vocab_size; i++) {
				esum += exp(ys[t][i]); 			// np.sum(np.exp(ys[t])) # probabilities for next chars
			}
			for(int i=0; i<vocab_size; i++) {
				ps[t][i] = exp(ys[t][i]) / (esum + 0.00001); 		// ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) - column vector
			}

			//printf("Wxh: %f\n", Wxh[0]);
			//print(Whh, h_size, vocab_size);

			//printf("p target t: %f\n", p[targets[t]][t]);
			sequence_loss += -log(ps[t][targets[t]]); // loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
			//printf("p target: %d : %d : %f\n", t, targets[t], sequence_loss);
		}

		//print(Why, vocab_size, h_size);
		//print(*hs, seq_length, h_size);
		//printf("pre backward:  %f\n", dWxh[0]);

		//printf("backward prop %f\n", Wxh[0]);
		// Model parameter initialization
		zerodmat(dWxh, h_size, vocab_size);
		zerodmat(dWhh, h_size, h_size);
		zerodmat(dWhy, vocab_size, h_size);
		zerodmat(dbh, h_size, 1);
		zerodmat(dby, vocab_size, 1);
		zerodmat(dhnext, h_size, 1);

	    // Backward prop ------------------------------------------------------------------
		for(int t=seq_length-1; t>-1; t--) { // index t counting down

			for(int i=0; i<vocab_size; i++) { 	// dy = np.copy(ps[t]) dy[targets[t]] -= 1     -   column vector
				dy[i] = ps[t][i];
			}
	        dy[targets[t]] -= 1.; // the current target (truth) is 1 for the current t (an 0 for all other t's)

			for(int i=0; i<vocab_size; i++) {
				for(int j=0; j<h_size; j++) {
					//dWhy[i*h_size+j] += dy[i] * hs[t][j];		// dWhy += np.dot(dy, hs[t].T)
					//if(dWhy[i*h_size+j]>5.0 && !isnan(dWhy[i*h_size+j]))
						//printf("bp: 0 dWhy: %f dy: %f hs: %f\n", dWhy[i*h_size+j], dy[i], hs[t][j]);
					dWhy[i*h_size+j] += dy[i] * hs[t][j];
				}
				dby[i] += dy[i];							// dby += dy
			}
			
			//watchdog(dh, h_size, 1, 50.0, "0 dh");
			//watchdog(dy, vocab_size, 1, 5.0, "dy");
			//watchdog(*hs, seq_length, h_size, 5.0, "dy");
			//watchdog(Why, vocab_size, h_size, 50.0, "dWhy", 0);
			//watchdog(dhnext, h_size, 1, 50.0, "dhnext");
	        
			//printf("backward dhnext 0: %f\n", dhnext[0]);
			zerodmat(dh, h_size, 1);
			for(int j=0; j<vocab_size; j++) {
				for(int i=0; i<h_size; i++) {
					dh[i] += Why[j*h_size+i] * dy[j];		// dh = np.dot(Why.T, dy) + dhnext # backprop into h
					//if(!isnan(dh[i]) && dh[i]>5.)
						//printf("bp dh: %f Why: %f dy: %f\n", dh[i], Why[j*h_size+i], dy[j]);
				}
			}
			double tmphs = 0.;
			for(int i=0; i<h_size; i++) {
				dh[i] += dhnext[i];
				tmphs += hs[t][i] * hs[t][i];	// dhraw = (1 - hs[t] * hs[t]) * dh
			}

			//print(Why, vocab_size, h_size);
			//printf("bp: dby: %f\n", dby[0]);
			//printf("backward hs: %f\n", hs[t][0]);
			//printf("backward dhraw: %f dh: %f\n", dhraw[0], dh[0]);
			//watchdog(dh, h_size, 1, 50.0, "1 dh", 0);
				
			for(int i=0; i<h_size; i++) {
				dhraw[i] = (1. - tmphs) * dh[i];	// dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
			}

			for(int i=0; i<h_size; i++) {
				dbh[i] += dhraw[i];							// dbh += dhraw 
			}

			//watchdog(dhraw, h_size, 1, 5.0, "dhraw");
	        
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<vocab_size; j++) {
					dWxh[i*vocab_size+j] += dhraw[i] * xs[t][j];		//  dWxh += np.dot(dhraw, xs[t].T)
				}
				for(int j=0; j<h_size; j++)
					if(t>0)
						dWhh[i*h_size+j] += dhraw[i] * hs[t-1][j];				// dWhh += np.dot(dhraw, hs[t-1].T)
					else
						dWhh[i*h_size+j] += dhraw[i] * hprev[j];				// dWhh += np.dot(dhraw, hs[t-1].T)
	        }

			//print(dWhh, h_size, vocab_size);
			//watchdog(dWxh, h_size, vocab_size, 50.0, "dWxh");
			//watchdog(dWhh, h_size, h_size, 5.0, "dWhh");
			//watchdog(dWhy, vocab_size, h_size, 5.0, "dWhy");
			//watchdog(dbh, h_size, 1, 5.0, "dbh");
			//watchdog(dby, vocab_size, 1, 5.0, "dby");
			
			//printf("backward dWhh: %f dhraw: %f\n", dWhh[0], dhraw[0]);
			zerodmat(dhnext, h_size, 1);
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<h_size; j++)
					dhnext[i] += limval(dWhh[j*h_size+i] * dhraw[j], -1., 1.); 		// dhnext = np.dot(Whh.T, dhraw)
			}
			//print(dWhh, h_size, h_size);
			//printf("backward dhnext 1: %f\n", dhnext[0]);
		}

		for(int i=0; i<h_size; i++)
			hprev[i] = hs[seq_length-1][i];				// hprev = hs[len(inputs) - 1]

		// parameter updates
		//printf("param update pre: %d  %f\n", nruns, dWxh[0]);
		// clip updates
		double cliplim = 5.0; // np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		double addnum = 1.e-8;
		for(int i=0; i<h_size; i++) {
			for(int j=0; j<vocab_size; j++) {
				dWxh[i*vocab_size+j] = limval(dWxh[i*vocab_size+j], -cliplim, cliplim);
			}
			for(int j=0; j<h_size; j++) {
				dWhh[i*h_size+j] = limval(dWhh[i*h_size+j], -cliplim, cliplim);
			}
			dbh[i] = limval(dbh[i], -cliplim, cliplim);
		}
		for(int i=0; i<vocab_size; i++) {
			for(int j=0; j<h_size; j++)
				dWhy[i*h_size+j] = limval(dWhy[i*h_size+j], -cliplim, cliplim);
			dby[i] = limval(dby[i], -cliplim, cliplim);
		}

		smooth_loss = smooth_loss * 0.999 + sequence_loss * 0.001;

		// Occasionally print loss information
	    if (nruns % print_interval == 0)
	    	printf("iter %d, loss: %f, smooth loss: %f", nruns, sequence_loss, smooth_loss);

		// perform parameter update with Adagrad
		// stochastic gradient descend: attempt a minibatch
		//printf("parameters at minibatch: h_size:%d nn:%d o_size:%d\n", h_size, nn, o_size);
		/*int bsize = (int) (h_size*vocab_size/2);
		for(int ib=0; ib<bsize; ib++) {
			int i = randint(0, h_size);
			int jn = randint(0, vocab_size);
			int jj = randint(0, h_size);
			Wxh[i*vocab_size+jn] += - learning_rate * dWxh[i*vocab_size+jn];
			Whh[i*h_size+jj] += - learning_rate * dWhh[i*h_size+jj];
			Why[i*h_size+jj] += - learning_rate * dWhy[i*h_size+jj];
		}
		bsize = (int) (h_size/2);
		for(int ib=0; ib<bsize; ib++) {
			int i = randint(0, h_size);
			int iv = randint(0, vocab_size);
			bh[i] += - learning_rate * dbh[i];
			by[iv] += - learning_rate * dby[iv];
		}*/

		/*for(int i=0; i<h_size; i++) {
			for(int j=0; j<vocab_size; j++) {
				Wxh[i*vocab_size+j] -= learning_rate * dWxh[i*vocab_size+j];
			}
			for(int j=0; j<h_size; j++) {
				Whh[i*h_size+j] -= learning_rate * dWhh[i*h_size+j];
			}
			bh[i] -= learning_rate * dbh[i];
		}
		for(int i=0; i<vocab_size; i++) {
			for(int j=0; j<h_size; j++) {
				Why[i*h_size+j] -= learning_rate * dWhy[i*h_size+j];
			}
			by[i] -= learning_rate * dby[i];
		}*/

		for(int i=0; i<h_size; i++) {
			for(int j=0; j<vocab_size; j++) {
				mWxh[i*vocab_size+j] += dWxh[i*vocab_size+j] * dWxh[i*vocab_size+j];
			}
			for(int j=0; j<h_size; j++) {
				mWhh[i*h_size+j] += dWhh[i*h_size+j] * dWhh[i*h_size+j];
			}
			mbh[i] += dbh[i] * dbh[i];
		}
		for(int i=0; i<vocab_size; i++) {
			for(int j=0; j<h_size; j++)
				mWhy[i*h_size+j] += dWhy[i*h_size+j] * dWhy[i*h_size+j];
			mby[i] += dby[i] * dby[i];
		}

		for(int i=0; i<h_size; i++) {
			for(int j=0; j<vocab_size; j++) {
				Wxh[i*vocab_size+j] -= learning_rate * dWxh[i*vocab_size+j] / sqrt(mWxh[i*vocab_size+j]+addnum);
			}
			for(int j=0; j<h_size; j++) {
				Whh[i*h_size+j] -= learning_rate * dWhh[i*h_size+j] / sqrt(mWhh[i*h_size+j]+addnum);
			}
			bh[i] -= learning_rate * dbh[i] / sqrt(mbh[i]+addnum);
		}
		for(int i=0; i<vocab_size; i++) {
			for(int j=0; j<h_size; j++)
				Why[i*h_size+j] -= learning_rate * dWhy[i*h_size+j] / sqrt(mWhy[i*h_size+j]+addnum);
			by[i] -= learning_rate * dby[i] / sqrt(mby[i]+addnum);
		}

		// sample results every once in a while ***********************************************************
		int n_sample = 200;
		if(nruns % print_interval == 0) {
			
			double xx[vocab_size]; 
			double hh[n_sample][h_size];		// hh->h_prev
			double yy[n_sample][vocab_size];
			double pp[n_sample][vocab_size];

			// sample_ix = sample(hprev, inputs[0], 1000) -> sample(h, seed_ix, n)
			zerodmat(xx, 1, vocab_size);
	        xx[inputs[0]] = 1; // Initialize first word of sample ('seed') as one-hot encoded vector
			int *ixes = malloc(n_sample * sizeof(int));
			ixes[inputs[0]] = 1; // seedix is inputs[0] sample_ix = sample(hprev, inputs[0], 1000)
			
			for(int tt=0; tt<n_sample; tt++) {

				for(int i=0; i<h_size; i++) {
					hh[tt][i] = 0.;
				}
				for(int i=0; i<h_size; i++) {
					for(int j=0; j<vocab_size; j++) {
						hh[tt][i] += Wxh[j*vocab_size+i] * xx[j];		// np.dot(Wxh, xs[t])
					}
				}

				double temphs[h_size];
				for(int i=0; i<h_size; i++)
					temphs[i] = 0.;
				for(int i=0; i<h_size; i++) {
					for(int j=0; j<h_size; j++) {
						if(tt>0)
							temphs[i] += Whh[j*h_size+i] * hh[tt-1][j];		// np.dot(Whh, hs[t-1])
						else
							temphs[i] += Whh[j*h_size+i] * hprev[j];		// np.dot(Whh, hs[t-1])
					}
				}

				for(int i=0; i<h_size; i++) {
					if(tt>0)
						hh[tt][i] += mytanh(temphs[i]+bh[i]); 	// hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
					else
						hprev[i] += mytanh(temphs[i]+bh[i]); 	// hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
				}

				for(int i=0; i<vocab_size; i++)
					yy[tt][i] = 0.;
				for(int i=0; i<vocab_size; i++) {
					for(int j=0; j<h_size; j++) {
						yy[tt][i] += Why[i*h_size+j] * hh[tt][j];		// np.dot(Why, hs[t])
					}
					yy[tt][i] += by[i];		// ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
				}

				double esum = 0.;
				for(int i=0; i<vocab_size; i++) {
					esum += exp(yy[tt][i]); 			// np.sum(np.exp(ys[t])) # probabilities for next chars
				}
				for(int i=0; i<vocab_size; i++) {
					pp[tt][i] = exp(yy[tt][i]) / esum; 		// ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
				}

				// Choose next char according to the distribution
				int ix = rnddchoice(sample_ix, vocab_size, *pp, vocab_size);
				zerodmat(xx, 1, vocab_size); // reset xx
				xx[ix] = 1.;
				ixes[tt] = ix;
			}
			//return ixes
			char *txt;
			txt = malloc(n_sample * sizeof *txt);
			ixset_to_char(ixes, n_sample, cset, txt); // txt = ''.join(ix_to_char[ix] for ix in sample_ix)
	        printf("----\n%s\n----\n", txt);
			free(txt);
		}

		// Prepare for next iteration
	    pos += seq_length; // walk through the text file in seq_length steps
		nruns++;
		//printf("after nruns update\n");
		//print(Wz, h_size, nn); // *********************************************************************
	}

	printf("Done!\n");

	free(Wxh);
	free(Whh);
	free(bh);

	free(Why);
	free(by);

	free(dWxh);
	free(dWhh);
	free(dbh);

	free(dWhy);
	free(dby);

	free(dhnext);

	//free(mWxh);
	//free(mWhh);
	//free(mbh);

	//free(mWhy);
	//free(mby);

	return 0;
}
