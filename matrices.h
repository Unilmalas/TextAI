/* current representation of a matrix  */
typedef struct Matrix{
	int rows;
	int columns;
	double **numbers;
} Matrix;

typedef int bool;
#define true 1
#define false 0

Matrix *identity(int length);
Matrix *inversion(Matrix *m);
Matrix *constructor(int r, int c);
int destroy_matrix(Matrix *m);
int print(Matrix *m);
int row_swap(Matrix *m, int a, int b);
int scalar_multiply(Matrix *m, double f);
int reduce(Matrix *m, int a, int b, float factor);
int equals(Matrix *m1, Matrix *m2);
Matrix *clone(Matrix *m);
Matrix *transpose(Matrix *m);
Matrix *randi_matrix(int rows, int columns, int modulo);
Matrix *randd_matrix(int rows, int columns);
Matrix *multiply(Matrix *m1, Matrix *m2);
int add(Matrix *m1, Matrix *m2);
int subtract(Matrix *, Matrix *);
Matrix *gram_schmidt(Matrix *);
double *projection(Matrix *, double *, int length);
int zero_vector(Matrix *);
Matrix *orthonormal_basis(Matrix *);
double determinant(Matrix *m);
Matrix *solved_aug_matrix(Matrix *);
void manual_entry(Matrix **m);
double *eigenvalues(Matrix *m);

int copyMatrix(Matrix *m0, Matrix *m1);
int setMatrixVal(Matrix *m, double a);
int setMatrixArr(Matrix *m, int r, int c, double arr[c][r]);
int setonevMatrix(Matrix *m, int i, int j, double a);
Matrix *fromarrMatrix(int r, int c, double arr[c][r]);
double maxMatrix(Matrix *m);
double sumMatrix(Matrix *m);
int negMatrix(Matrix *m);
Matrix *hadamardMatrix(Matrix *m1, Matrix *m2);
int expMatrix(Matrix *m);
int logMatrix(Matrix *m);

int sigMatrix(Matrix *m, bool deriv);
int tanhMatrix(Matrix *m, bool deriv);
int softMatrix(Matrix *m);
