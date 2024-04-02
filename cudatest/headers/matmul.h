float** make_matrix(int r, int c);

void delete_matrix(float **mat, int r, int c);

float **set(float **mat, float val, int r, int c);

void to_device(float **dst, float** src, int r, int c);

void print(float **mat, int r, int c);

///////////

void add(float **A, float **B, float **C, int r, int c);

