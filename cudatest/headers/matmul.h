float* make_matrix(int r, int c);

void delete_matrix(float *mat);

void set(float *mat, float val, int r, int c);

float* to_device(float *src, int r, int c);

void add(float *A, float *B, float *C, int r, int c);

void cuda_print(float *mat, int r, int c);

void print(float *mat, int r, int c);

float* to_host(float *src, int r, int c);
