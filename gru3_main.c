// GRU with Minimal Gated Units
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
//#include<err.h>
#include <time.h>

#define SUCC 1
#define FAIL -1

#define MAXVOCS 100	// max vocabulary size

int ae_load_file_to_memory(const char *filename, char **result) { 
	int size = 0;
	FILE *f = fopen(filename, "rb");
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

static int fwdprop(int tt, long nn, long h_size, long o_size, long seq_length,
				double *Wz, double *Wr, double *Wh, double *Wy,
				double *bz, double *br, double *bh, double *by,
				double *Uz, double *Ur, double *Uh,
				double *xx, double *zz, double *rr, double *hh, double *hh_hat,
				double *hprev, double *yy) {
	// Calculate update and reset gates
	//z = sigmoid(np.dot(Wz, x) + np.dot(Uz, h) + bz)
	//r = sigmoid(np.dot(Wr, x) + np.dot(Ur, h) + br)
	for(int i=0; i<h_size; i++) { // nrows
		double temp_wzx = 0.;
		double temp_uzh = 0.;
		double temp_wrx = 0.;
		double temp_urh = 0.;
		for(int j=0; j<nn; j++) { // ncols
			temp_wzx += Wz[i*nn+j] * xx[j]; 		// np.dot(Wz, x[t]) -> Wz[i*ncols+j]
			temp_wrx += Wr[i*nn+j] * xx[j]; 		// np.dot(Wr, x[t])
		}
		for(int j=0; j<h_size; j++) {
			temp_uzh += Uz[i*h_size+j] * hprev[j*seq_length+seq_length-1]; 	// np.dot(Uz, h[t-1])
			temp_urh += Ur[i*h_size+j] * hprev[j*seq_length+seq_length-1]; 	// np.dot(Ur, h[t-1]) + br)
		}
		zz[i] = sigmoid(temp_wzx + temp_uzh + bz[i]);
		rr[i] = sigmoid(temp_wrx + temp_urh + br[i]);
	}
    
	// Calculate hidden units
	//h_hat = tanh(np.dot(Wh, x) + np.dot(Uh, np.multiply(r, h)) + bh)
	//h = np.multiply(z, h) + np.multiply((1 - z), h_hat)
	double *temp_rh;
	temp_rh = malloc(h_size * sizeof *temp_rh);
	for(int i=0; i<h_size; i++) {
		temp_rh[i] = rr[i] * hprev[i*seq_length+seq_length-1]; // np.multiply(r[t], h[t-1]) ********
	}
	for(int i=0; i<h_size; i++) {
		double temp_whx = 0.;
		double temp_uhr = 0.;
		for(int j=0; j<nn; j++) {
			temp_whx += Wh[i*nn+j] * xx[j];			// np.dot(Wh, x[t])
		}
		for(int j=0; j<h_size; j++) {
			temp_uhr += Uh[i*h_size+j] * temp_rh[j];	// np.dot(Uh, np.multiply(r[t], h[t-1]))
		}
		hh_hat[i] = mytanh(temp_whx + temp_uhr + bh[i]);
	}
	for(int i=0; i<h_size; i++)
		hh[i] = zz[i] * hprev[i*seq_length+seq_length-1] + (1. - zz[i]) * hh_hat[i];	// np.multiply(z[t], h[t-1]) + np.multiply((1 - z[t]), h_hat[t])
    
	// Regular output unit
	//y = np.dot(Wy, h) + by
	//printf("Wy: %d\n", Wy[0]);
	for(int i=0; i<o_size; i++) {
		double temp_wy = 0.;
		for(int j=0; j<h_size; j++) {
			temp_wy += Wy[i*h_size+j] * hh[j];	// np.dot(Wy, h[t])
		}
		yy[i] = temp_wy + by[i];
	}
    
	return SUCC;
}


int main (int argc, char **argv) {
	// init random generator (only once!)
	srand(time(NULL));
	// load text file into memory
	char *data; // corpus stored here
	int fres = ae_load_file_to_memory("input.txt", &data);
	long data_size = strlen(data);
	char *cset = (char *)malloc(MAXVOCS+1);	// create character map
	long vocab_size = charset(data, cset);
	printf("%d chars found: %s in file of length %d\n", strlen(cset), cset, data_size);
	
	int seq_length = 5; // Longer sequence lengths allow for lengthier latent dependencies to be trained.
	double learning_rate = 0.1; // very sensitive, improvement after change from 0.1 to 0.01
	
	// Hidden size is set to vocab_size, assuming that level of abstractness is approximately proportional to vocab_size
	// (but can be set to any other value).
	long nn = vocab_size;
	long h_size = vocab_size;
	long o_size = vocab_size;

	int sample_ix[vocab_size];
	for(int i=0; i<vocab_size; i++)
		sample_ix[i] = i;

	double smooth_loss = 0.;

	long pos = 0;
	long nruns = 0;

	double *Wz, *Uz, *bz;
	double *Wr, *Ur, *br;
	double *Wh, *Uh, *bh;
	double *Wy, *by;
	
	Wz = malloc(h_size * nn * sizeof *Wz);
	Uz = malloc(h_size * h_size * sizeof *Uz);
	bz = malloc(h_size * 1 * sizeof *bz);

	Wr = malloc(h_size * nn * sizeof *Wr);
	Ur = malloc(h_size * h_size * sizeof *Ur);
	br = malloc(h_size * 1 * sizeof *br);

	Wh = malloc(h_size * nn * sizeof *Wh);
	Uh = malloc(h_size * h_size * sizeof *Uh);
	bh = malloc(h_size * 1 * sizeof *bh);

	Wy = malloc(o_size * h_size * sizeof *Wy);
	by = malloc(o_size * 1 * sizeof *by);

	// Model parameter initialization - 0.2 seems to converge faster than 0.1
	randdmat(Wz, h_size, nn, 0.1);
	randdmat(Uz, h_size, h_size, 0.1);
	zerodmat(bz, h_size, 1);

	randdmat(Wr, h_size, nn, 0.1);
	randdmat(Ur, h_size, h_size, 0.1);
	zerodmat(br, h_size, 1);

	randdmat(Wh, h_size, nn, 0.1);
	randdmat(Uh, h_size, h_size, 0.1);
	zerodmat(bh, h_size, 1);

	randdmat(Wy, o_size, h_size, 0.1);
	zerodmat(by, o_size, 1);

    // Initialize variables
    double x[nn][seq_length]; // [vocab_size][len of inputs = seq_length = t] 
	double z[h_size][seq_length];
	double r[h_size][seq_length];
	double h_hat[h_size][seq_length];
	double h[h_size][seq_length];
	double y[o_size][seq_length];
	double p[o_size][seq_length];
	double hprev[h_size][seq_length]; // init hprev with h[t-1]; h[t] is then calculated in the forward propagation step

    // Parameter gradient initialization
	double *dWz, *dUz, *dbz;
	double *dWr, *dUr, *dbr;
	double *dWh, *dUh, *dbh;
	double *dWy, *dby;
	double *dhnext;
	double *dy, *dh, *dh_hat, *dh_hat_l;
	double *drhp, *dr, *dr_l, *dz_l;
	double *dz, *dh_fz_inner, *dh_fz, *dh_fhh, *dh_fr;

	double tmpsig;
	
	dWz = malloc(h_size * nn * sizeof *dWz);
	dUz = malloc(h_size * h_size * sizeof *dUz);
	dbz = malloc(h_size * 1 * sizeof *dbz);

	dWr = malloc(h_size * nn * sizeof *dWr);
	dUr = malloc(h_size * h_size * sizeof *dUr);
	dbr = malloc(h_size * 1 * sizeof *dbr);

	dWh = malloc(h_size * nn * sizeof *dWh);
	dUh = malloc(h_size * h_size * sizeof *dUh);
	dbh = malloc(h_size * 1 * sizeof *dbh);

	dWy = malloc(o_size * h_size * sizeof *dWy);
	dby = malloc(o_size * 1 * sizeof *dby);

	dhnext = malloc(h_size * 1 * sizeof *dhnext);
	dy = malloc(o_size * 1 * sizeof *dy);
	dh = malloc(h_size * 1 * sizeof *dh);
	dh_hat = malloc(h_size * 1 * sizeof *dh_hat);
	dh_hat_l = malloc(h_size * 1 * sizeof *dh_hat_l);
	drhp = malloc(h_size * 1 * sizeof *drhp);
	dr = malloc(h_size * 1 * sizeof *dr);
	dr_l = malloc(h_size * 1 * sizeof *dr_l);
	dz_l = malloc(h_size * 1 * sizeof *dz_l);

	dz = malloc(h_size * 1 * sizeof *dz);
	dh_fz_inner = malloc(h_size * 1 * sizeof *dh_fz_inner);
	dh_fz = malloc(h_size * 1 * sizeof *dh_fz);
	dh_fhh = malloc(h_size * 1 * sizeof *dh_fhh);
	dh_fr = malloc(h_size * 1 * sizeof *dh_fr);

    double sequence_loss = 0.;
	long print_interval = 100;

	smooth_loss = -log(1.0/vocab_size)*seq_length;

	// previous gradients
	double *pdWz, *pdUz, *pdbz;
	double *pdWr, *pdUr, *pdbr;
	double *pdWh, *pdUh, *pdbh;
	double *pdWy, *pdby;
	pdWz = malloc(h_size * nn * sizeof *pdWz);
	pdUz = malloc(h_size * h_size * sizeof *pdUz);
	pdbz = malloc(h_size * 1 * sizeof *pdbz);

	pdWr = malloc(h_size * nn * sizeof *pdWr);
	pdUr = malloc(h_size * h_size * sizeof *pdUr);
	pdbr = malloc(h_size * 1 * sizeof *pdbr);

	pdWh = malloc(h_size * nn * sizeof *pdWh);
	pdUh = malloc(h_size * h_size * sizeof *pdUh);
	pdbh = malloc(h_size * 1 * sizeof *pdbh);

	pdWy = malloc(o_size * h_size * sizeof *pdWy);
	pdby = malloc(o_size * 1 * sizeof *pdby);

	// gradient memory initialization
	zerodmat(pdWz, h_size, nn);
	zerodmat(pdUz, h_size, h_size);
	zerodmat(pdbz, h_size, 1);

	zerodmat(pdWr, h_size, nn);
	zerodmat(pdUr, h_size, h_size);
	zerodmat(pdbr, h_size, 1);

	zerodmat(pdWh, h_size, nn);
	zerodmat(pdUh, h_size, h_size);
	zerodmat(pdbh, h_size, 1);

	zerodmat(pdWy, o_size, h_size);
	zerodmat(pdby, o_size, 1);

	//print(Wy, h_size, nn);

	// while true
	//while(nruns < (data_size - seq_length)) { // step through entire text
	while (nruns < 1000) {
	    // Reset memory if appropriate - reset position to 0
	    if(pos + seq_length + 1 >= data_size || nruns==0 ) {
	        //hprev = np.zeros((h_size, 1)) # np.zeros(shape) -> (vocab_size, 1) = (rows, cols)
			zerodmat(*hprev, h_size, seq_length);
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
			//printf("inputs: %d\n", inputs[i]);
		}

		// Model parameter initialization
		zerodmat(dWz, h_size, nn);
		zerodmat(dUz, h_size, h_size);
		zerodmat(dbz, h_size, 1);

		zerodmat(dWr, h_size, nn);
		zerodmat(dUr, h_size, h_size);
		zerodmat(dbr, h_size, 1);

		zerodmat(dWh, h_size, nn);
		zerodmat(dUh, h_size, h_size);
		zerodmat(dbh, h_size, 1);

		zerodmat(dWy, o_size, h_size);
		zerodmat(dby, o_size, 1);

		sequence_loss = 0.;

		// Get gradients for current model based on input and target sequences
	    // Forward prop ------------------------------------------------------------------------
	    for(int t=0; t<seq_length; t++) { // lossFun(inputs, targets, hprev)
	        // Set up one-hot encoded input
	        //x[t] = np.zeros((vocab_size, 1)) // for each time step t a one-hot over vocabulary size - init with all 0
			//memset(x[t], 0, sizeof(x)); // for each time step t a one-hot over vocabulary size - init with all 0
			for(int i=0; i<nn; i++) { // nrows
				x[i][t] = 0.;
			}
	        x[inputs[t]][t] = 1.; //x[t][inputs[t]] = 1; // set t'th input to one (current word in one-hot)

			//printv(*x, nn, seq_length, t, 1);

	        // Calculate update and reset gates
	        //z[t] = sigmoid(np.dot(Wz, x[t]) + np.dot(Uz, h[t-1]) + bz) // update gate
	        //r[t] = sigmoid(np.dot(Wr, x[t]) + np.dot(Ur, h[t-1]) + br) // reset gate
			for(int i=0; i<h_size; i++) { // nrows
				double temp_wzx = 0.;
				double temp_uzh = 0.;
				double temp_wrx = 0.;
				double temp_urh = 0.;
				for(int j=0; j<nn; j++) { // ncols
					temp_wzx += Wz[i*nn+j] * x[j][t]; 		// np.dot(Wz, x[t]) -> Wz[i*ncols+j]
					temp_wrx += Wr[i*nn+j] * x[j][t]; 		// np.dot(Wr, x[t])
				}
				for(int j=0; j<h_size; j++) {
					if (t>0) {
						temp_uzh += Uz[i*h_size+j] * h[j][t-1]; 	// np.dot(Uz, h[t-1])
						temp_urh += Ur[i*h_size+j] * h[j][t-1]; 	// np.dot(Ur, h[t-1]) + br)
					} else {
						temp_uzh += Uz[i*h_size+j] * hprev[j][0]; 	// np.dot(Uz, h[t-1])
						temp_urh += Ur[i*h_size+j] * hprev[j][0]; 	// np.dot(Ur, h[t-1]) + br) - should be hprev
					}
				}
				z[i][t] = sigmoid(temp_wzx + temp_uzh + bz[i]);
				r[i][t] = sigmoid(temp_wrx + temp_urh + br[i]);
			}

			//printv(*r, h_size, seq_length, t, 1);
	        
	        // Calculate hidden units
	        //h_hat[t] = tanh(np.dot(Wh, x[t]) + np.dot(Uh, np.multiply(r[t], h[t-1])) + bh)
	        //h[t] = np.multiply(z[t], h[t-1]) + np.multiply((1 - z[t]), h_hat[t]) // sometimes denoted s[t]
			double *temp_rh;
			temp_rh = malloc(h_size * sizeof *temp_rh);
			for(int i=0; i<h_size; i++) {
				if (t>0)
					temp_rh[i] = r[i][t] * h[i][t-1]; // np.multiply(r[t], h[t-1])
				else
					temp_rh[i] = r[i][t] * hprev[i][0]; // np.multiply(r[t], h[t-1])
			}
			for(int i=0; i<h_size; i++) {
				double temp_whx = 0.;
				double temp_uhr = 0.;
				for(int j=0; j<nn; j++) {
					temp_whx += Wh[i*nn+j] * x[j][t];		// np.dot(Wh, x[t])
				}
				for(int j=0; j<h_size; j++) {
					temp_uhr += Uh[i*h_size+j] * temp_rh[j];	// np.dot(Uh, np.multiply(r[t], h[t-1]))
				}
				h_hat[i][t] = mytanh(temp_whx + temp_uhr + bh[i]);
			}
			//printv(*h_hat, h_size, seq_length, t, 1); // careful: sensitive to tanh-approx (and x-range-checks there!)
			for(int i=0; i<h_size; i++) {
				double temp_zh = 0.;
				if (t>0)
					temp_zh = z[i][t] * h[i][t-1];				// np.multiply(z[t], h[t-1])
				else
					temp_zh = z[i][t] * hprev[i][0];				// np.multiply(z[t], h[t-1])		
				//printf("h fwd: t: %d  %f <- %f\n", t, h[i][t], hprev[i][0]);
				h[i][t] = temp_zh + (1. - z[i][t]) * h_hat[i][t];			// np.multiply((1 - z[t]), h_hat[t])
			}
			//printv(*h, h_size, seq_length, t, 1); 
	        
	        // Regular output unit
	        //y[t] = np.dot(Wy, h[t]) + by
			for(int i=0; i<o_size; i++) {
				double temp_wy = 0.;
				for(int j=0; j<h_size; j++) {
					temp_wy += Wy[i*h_size+j] * h[j][t];	// np.dot(Wy, h[t])
				}
				y[i][t] = temp_wy + by[i];
			}
	        
	        // Probability distribution
	        //p[t] = softmax(y[t])
			//softmax(y[t], h_size, p[t]); 
			softmax2(*y, o_size, seq_length, t, *p, o_size, seq_length, t);
	        
	        // Cross-entropy loss
	        // targets[t]'s entry of p[t]: since the output is 0 or 1 only one entry contributes (and yj*log(pj) becomes just log(pj))
	        // dict p: or time t (=one-hot position), whats the corresponsing target value?
			//printf("p target t: %f\n", p[targets[t]][t]);
			sequence_loss += -log(p[targets[t]][t]); // loss = -np.sum(np.log(p[t][targets[t]])) -> loss = -np.log(p[t][targets[t]])
		}

	    // Backward prop ------------------------------------------------------------------
		for(int t=seq_length-1; t>-1; t--) { // index t counting down

	        // ∂loss/∂y
	        //dy = np.copy(p[t]) // copy output
			for(int i=0; i<o_size; i++) {
				dy[i] = p[i][t];
			}
	        dy[targets[t]] -= 1.; // the current target (truth) is 1 for the current t (an 0 for all other t's)
	        
	        // ∂loss/∂Wy and ∂loss/∂by
	        //dWy += np.dot(dy, h[t].T) // weight updates: Wy -> Wy - etha * d loss / dWy -> dWy += etha * d loss / dWy
	        //dby += dy // weight for bias is just 1
			// numpy product of two vectors: [[1],[2]] * [[1, 2]] =[[1, 2],[2, 4]]
			for(int i=0; i<o_size; i++) {
				for(int j=0; j<h_size; j++) {
					dWy[i*h_size+j] += dy[i] * h[j][t];		// dWy += np.dot(dy, h[t].T)
				}
				dby[i] += dy[i];							// dby += dy
			}
	        
	        // Intermediary derivatives
			zerodmat(dh, h_size, 1);
			for(int i=0; i<o_size; i++) {
				for(int j=0; j<h_size; j++) {
					dh[j] += Wy[j*h_size+i] * dy[i];		// dh = np.dot(Wy.T, dy) + dhnext
				}
				dby[i] += dhnext[i];
			}
			for(int i=0; i<h_size; i++) {
				dh_hat[i] = dh[i] * (1. - z[i][t]);			// dh_hat = np.multiply(dh, (1 - z[t]))
			}
			for(int i=0; i<h_size; i++) {
				double tmpth = mytanh( h_hat[i][t] ); // sometimes pow(..,2) is used, this makes it more numerically stable
				dh_hat_l[i] = dh_hat[i] * (1. - tmpth * tmpth);	// dh_hat_l = dh_hat * tanh(h_hat[t], deriv=True) 
			}
	        
	        // ∂loss/∂Wh, ∂loss/∂Uh and ∂loss/∂bh
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<nn; j++) {
					dWh[i*nn+j] += dh_hat_l[i] * x[j][t];		// dWh += np.dot(dh_hat_l, x[t].T)
				}
			}
			double temp_rh[h_size];
			for(int i=0; i<h_size; i++) {
				if (t>0)
					temp_rh[i] = r[i][t] * h[i][t-1];			// np.multiply(r[t], h[t-1]).T)
				else
					temp_rh[i] = r[i][t] * hprev[i][0];			// np.multiply(r[t], h[t-1]).T)
			}				
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<h_size; j++) {
					dUh[i*h_size+j] += dh_hat_l[i] * temp_rh[j];	// dUh += np.dot(dh_hat_l, np.multiply(r[t], h[t-1]).T)
				}
			}	

			// dbh += dh_hat_l // weight for bias is just 1
			for(int i=0; i<h_size; i++) {
	        	dbh[i] += dh_hat_l[i];						// dbh += dh_hat_l // weight for bias is just 1
	        }

	        // Intermediary derivatives
			for(int i=0; i<h_size; i++) {
				double temp_uhdh = 0.;
				for(int j=0; j<h_size; j++) {
					temp_uhdh += Uh[j*h_size+i] * dh_hat_l[j]; // drhp = np.dot(Uh.T, dh_hat_l)
				}
				drhp[i] = temp_uhdh;
			}
			for(int i=0; i<h_size; i++) {
				if (t>0)
					dr[i] = drhp[i] * h[i][t-1];				// dr = np.multiply(drhp, h[t-1])
				else
					dr[i] = drhp[i] * hprev[i][0];				// dr = np.multiply(drhp, h[t-1])
				tmpsig = sigmoid( r[i][t] );
				dr_l[i] = dr[i] * tmpsig * (1. - tmpsig); 		// dr_l = dr * sigmoid(r[t], deriv=True)
			}
	        
	        // ∂loss/∂Wr, ∂loss/∂Ur and ∂loss/∂br
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<nn; j++)
					dWr[i*nn+j] += dr_l[i] * x[j][t];				// dWr += np.dot(dr_l, x[t].T)
				for(int j=0; j<h_size; j++) {
					if (t>1)
						dUr[i*h_size+j] += dr_l[i] * h[j][t-1];		// dUr += np.dot(dr_l, h[t-1].T)
					else
						dUr[i*h_size+j] += dr_l[i] * hprev[j][0];	// dUr += np.dot(dr_l, h[t-1].T)
				}
				dbr[i] = dr_l[i];									// dbr += dr_l
	        }

	        // Intermediary derivatives
			for(int i=0; i<h_size; i++) {
				if (t>0)
					dz[i] = dh[i] * ( h[i][t-1] - h_hat[i][t] );	// dz = np.multiply(dh, h[t-1] - h_hat[t])
				else
					dz[i] = dh[i] * ( hprev[i][0] - h_hat[i][t] );	// dz = np.multiply(dh, h[t-1] - h_hat[t])
				tmpsig = sigmoid( z[i][t] );
				dz_l[i] = dz[i] * tmpsig * (1. - tmpsig);			// dz_l = dz * sigmoid(z[t], deriv=True)
			}

			//printf("param update pre0: t: %d  %f\n", t, dWz[0]);
			//printf("dz_l pre0: t: %d  %f\n", t, dz_l[0]);
	        
	        // ∂loss/∂Wz, ∂loss/∂Uz and ∂loss/∂bz
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<nn; j++)
					dWz[i*nn+j] = dz_l[i] * x[j][t];				// dWz += np.dot(dz_l, x[t].T)
				for(int j=0; j<h_size; j++) {
					if (t>0)
						dUz[i*h_size+j] = dz_l[i] * h[j][t-1];		// dUz += np.dot(dz_l, h[t-1].T)
					else
						dUz[i*h_size+j] = dz_l[i] * hprev[j][0];	// dUz += np.dot(dz_l, h[t-1].T)
				}
				dbz[i] = dz_l[i];									// dbz += dz_l
			}

			//printf("param update pre1: t: %d  %f\n", t, dWz[0]);
	        
	        // All influences of previous layer to loss
	        //dh_fz_inner = np.dot(Uz.T, dz_l) 
	        //dh_fz = np.multiply(dh, z[t]) 
	        //dh_fhh = np.multiply(drhp, r[t]) 
	        //dh_fr = np.dot(Ur.T, dr_l) 
			zerodmat(dh_fz_inner, h_size, 1);
			zerodmat(dh_fr, h_size, 1);
			for(int i=0; i<h_size; i++) {
				for(int j=0; j<h_size; j++) {
					dh_fz_inner[i] += Uz[j*h_size+i] * dz_l[j];	// np.dot(Uz.T, dz_l)
					dh_fr[i] += Ur[j*h_size+i] * dr_l[j];		// np.dot(Ur.T, dr_l)
				}
				dh_fz[i] = dh[i] * z[i][t];						// np.multiply(dh, z[t])
				dh_fhh[i] = drhp[i] * r[i][t];					// np.multiply(drhp, r[t])
	        }

	        // ∂loss/∂h𝑡₋₁
	        //dhnext = dh_fz_inner + dh_fz + dh_fhh + dh_fr
			for(int i=0; i<h_size; i++) {
				dhnext[i] = dh_fz_inner[i] + dh_fz[i] + dh_fhh[i] + dh_fr[i];
			}
			
			for(int i=0; i<h_size; i++)
				hprev[i][t] = h[i][seq_length-1];				// hprev = h[len(inputs) - 1]
		}

		// parameter updates
		//printf("param update pre: %d  %f\n", nruns, dWz[0]);
		// clip updates
		double cliplim = 5.0;
		double addnum = 1.e-8;
		for(int i=0; i<h_size; i++) {
			for(int j=0; j<nn; j++) {
				dWz[i*nn+j] = limval(dWz[i*nn+j], -cliplim, cliplim);
				dWr[i*nn+j] = limval(dWr[i*nn+j], -cliplim, cliplim);
				dWh[i*nn+j] = limval(dWh[i*nn+j], -cliplim, cliplim);
			}
			for(int j=0; j<h_size; j++) {
				dUz[i*h_size+j] = limval(dUz[i*h_size+j], -cliplim, cliplim);
				dUr[i*h_size+j] = limval(dUr[i*h_size+j], -cliplim, cliplim);
				dUh[i*h_size+j] = limval(dUh[i*h_size+j], -cliplim, cliplim);
			}
			dbz[i] = limval(dbz[i], -cliplim, cliplim);
			dbr[i] = limval(dbr[i], -cliplim, cliplim);
			dbh[i] = limval(dbh[i], -cliplim, cliplim);
			
		}
		for(int i=0; i<o_size; i++)
			for(int j=0; j<h_size; j++) {
				dWy[i*h_size+j] = limval(dWy[i*h_size+j], -cliplim, cliplim);
			dby[i] = limval(dby[i], -cliplim, cliplim);
		}

		for(int i=0; i<h_size; i++) {
			for(int j=0; j<nn; j++) {
				pdWz[i*nn+j] += dWz[i*nn+j] * dWz[i*nn+j];
				pdWr[i*nn+j] += dWr[i*nn+j] * dWr[i*nn+j];
				pdWh[i*nn+j] += dWh[i*nn+j] * dWh[i*nn+j];
			}
			for(int j=0; j<h_size; j++) {
				pdUz[i*h_size+j] += dUz[i*h_size+j] * dUz[i*h_size+j];
				pdUr[i*h_size+j] += dUr[i*h_size+j] * dUr[i*h_size+j];
				pdUh[i*h_size+j] += dUh[i*h_size+j] * dUh[i*h_size+j];
			}
			pdbz[i] += dbz[i] * dbz[i];
			pdbr[i] += dbr[i] * dbr[i];
			pdbh[i] += dbh[i] * dbh[i];
		}
		for(int i=0; i<o_size; i++) {
			for(int j=0; j<h_size; j++)
				pdWy[i*h_size+j] += dWy[i*h_size+j] * dWy[i*h_size+j];
			pdby[i] += dby[i]*dby[i];
		}

		// param += -learningrate * dparam / sqrt(dparam^2 + 1e-8)
		for(int i=0; i<h_size; i++) {
			for(int j=0; j<nn; j++) {
				Wz[i*nn+j] += - learning_rate * dWz[i*nn+j] / (sqrt( pdWz[i*nn+j] ) + addnum);
				Wr[i*nn+j] += - learning_rate * dWr[i*nn+j] / (sqrt( pdWr[i*nn+j] ) + addnum);
				Wh[i*nn+j] += - learning_rate * dWh[i*nn+j] / (sqrt( pdWh[i*nn+j] ) + addnum);
			}
			for(int j=0; j<h_size; j++) {
				Uz[i*h_size+j] += - learning_rate * dUz[i*h_size+j] / (sqrt( pdUz[i*h_size+j] ) + addnum);
				Ur[i*h_size+j] += - learning_rate * dUr[i*h_size+j] / (sqrt( pdUr[i*h_size+j] ) + addnum);
				Uh[i*h_size+j] += - learning_rate * dUh[i*h_size+j] / (sqrt( pdUh[i*h_size+j] ) + addnum);
			}
			bz[i] += - learning_rate * dbz[i] / (sqrt( pdbz[i] ) + addnum);
			br[i] += - learning_rate * dbr[i] / (sqrt( pdbr[i] ) + addnum);
			bh[i] += - learning_rate * dbh[i] / (sqrt( pdbh[i] ) + addnum);
		}
		for(int i=0; i<o_size; i++) {
			for(int j=0; j<h_size; j++)
				Wy[i*h_size+j] += - learning_rate * dWy[i*h_size+j] / (sqrt( pdWy[i*h_size+j] ) + addnum);
			by[i] += - learning_rate * dby[i] / (sqrt( pdby[i] ) + addnum);
		}
		
		smooth_loss = smooth_loss * 0.999 + sequence_loss * 0.001;
		// Occasionally print loss information
	    if (nruns % print_interval == 0)
	    	printf("iter %d, loss: %f, smooth loss: %f", nruns, sequence_loss, smooth_loss);
		//print(Wz, h_size, nn);
		//printf("param update post: %d  %f\n", nruns, dWz[0]);

		// sample results every once in a while ***********************************************************
		// vars for sample forward propagation
		if(nruns % print_interval == 0) {
			
			double xx[nn]; 
			double zz[h_size];
			double rr[h_size];
			double hh_hat[h_size]; 
			double hh[h_size];		// hh->h_prev
			double yy[o_size];
			double pp[o_size];

			int n_sample = 400;

			// sample_ix = sample(hprev, inputs[0], 1000) -> sample(h, seed_ix, n)
			zerodmat(xx, nn, 1);
	        xx[inputs[0]] = 1; // Initialize first word of sample ('seed') as one-hot encoded vector
			int *ixes = malloc(n_sample * sizeof(int));
			ixes[inputs[0]] = 1; // seedix is inputs[0] sample_ix = sample(hprev, inputs[0], 1000)
			
			for(int tt=0; tt<n_sample; tt++) {
				fwdprop(tt, nn, h_size, o_size, seq_length,
					Wz, Wr, Wh, Wy,
					bz, br, bh, by,
					Uz, Ur, Uh,
					xx, zz, rr, hh, hh_hat,
					*hprev, yy); 
				// Probability distribution
				//p = softmax(y)
				//softmax(yy, h_size, pp);
				softmax2(yy, o_size, 1, 0, pp, o_size, 1, 0);

				// Choose next char according to the distribution
				//ix = np.random.choice(range(vocab_size), p=p.ravel()) // ravel returns flattened array, P are the probabilities for choice
				int ix = rnddchoice(sample_ix, vocab_size, pp, vocab_size);
				//x = np.zeros((vocab_size, 1))
				//x[ix] = 1
				//ixes.append(ix)
				zerodmat(xx, 1, nn); // reset xx
				xx[ix] = 1.;
				ixes[tt] = ix;
				//for(int i=0; i<vocab_size; i++)
				//printf("%d, ", sample_ix[i]);
			}
			//return ixes
			char *txt;
			txt = malloc(n_sample * sizeof *txt);
			ixset_to_char(ixes, n_sample, cset, txt); // txt = ''.join(ix_to_char[ix] for ix in sample_ix)
	        printf("----\n%s\n----\n", txt);
		}

		// Prepare for next iteration
	    pos += seq_length; // walk through the text file in seq_length steps
		nruns++;
		//printf("after nruns update\n");
		//print(Wz, h_size, nn); // *********************************************************************
	}

	//printf("final:\n");
	//print(Wy, h_size, nn);
	printf("Done!\n");

	free(Wz);
	free(Uz);
	free(bz);

	free(Wr);
	free(Ur);
	free(br);

	free(Wh);
	free(Uh);
	free(bh);

	free(Wy);
	free(by);

	free(dWz);
	free(dUz);
	free(dbz);

	free(dWr);
	free(dUr);
	free(dbr);

	free(dWh);
	free(dUh);
	free(dbh);

	free(dWy);
	free(dby);

	free(pdWz);
	free(pdUz);
	free(pdbz);

	free(pdWr);
	free(pdUr);
	free(pdbr);

	free(pdWh);
	free(pdUh);
	free(pdbh);

	free(pdWy);
	free(pdby);

	free(dhnext);
	free(dy);
	free(dh);
	free(dh_hat);
	free(dh_hat_l);
	free(drhp);
	free(dr);
	free(dr_l);
	free(dz_l);

	free(dz);
	free(dh_fz_inner);
	free(dh_fz);
	free(dh_fhh);
	free(dh_fr);

	return 0;
}
