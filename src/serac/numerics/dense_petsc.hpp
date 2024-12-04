
#include <slepcsvd.h>
#include <slepcbv.h>
#include <vector>

struct DenseVec;

struct DenseMat
{
  DenseMat(const Mat& a) : A(a) {}
  
  DenseMat(const DenseMat& a) {
    MatDuplicate(a.A, MAT_COPY_VALUES, &A);
    MatCopy(a.A, A, SAME_NONZERO_PATTERN);
  }

  DenseMat& operator=(const DenseMat& a) {
    MatCopy(a.A, A, SAME_NONZERO_PATTERN);
    return *this;
  }

  ~DenseMat() { MatDestroy(&A); }

  auto size() const {
    int isize; int jsize;
    MatGetSize(A, &isize, &jsize);
    return std::make_pair(isize,jsize);
  }

  DenseVec operator*(const DenseVec& v) const;
  DenseVec solve(const DenseVec& v) const;
  DenseMat PtAP(const DenseMat& P) const;

  void print(std::string first = "") const {
    if (first.size()) {
      std::cout << first << ": " ;
    }
    MatView(A, PETSC_VIEWER_STDOUT_SELF);
  }

  Mat A;
};

DenseMat inverse(const DenseMat& a)
{
  Mat inv; MatDuplicate(a.A, MAT_COPY_VALUES, &inv);
  MatSeqDenseInvert(inv);
  return inv;
}

struct DenseVec
{
  DenseVec(const Vec& vin) : v(vin) {}

  DenseVec(const DenseVec& vin) {
    VecDuplicate(vin.v, &v);
    VecCopy(vin.v, v);
  }

  DenseVec& operator= (const DenseVec& vin) {
    VecCopy(vin.v, v);
    return *this;
  }

  DenseVec& operator= (const double val) {
    VecSet(v, val);
    return *this;
  }

  DenseVec(size_t size) { VecCreateSeq(PETSC_COMM_SELF, static_cast<int>(size), &v); }
  DenseVec(int size) { VecCreateSeq(PETSC_COMM_SELF, size, &v); }

  DenseVec(const std::vector<double> vin) {
    const auto sz = vin.size();
    std::vector<int> allints(sz);
    for (size_t i=0; i < sz; ++i) {
      allints[i] = static_cast<int>(i);
    }
    int sz_int = static_cast<int>(sz);
    VecCreateSeq(PETSC_COMM_SELF, sz_int, &v);
    VecSetValues(v, sz_int, &allints[0], &vin[0], INSERT_VALUES);
  }
  
  ~DenseVec() { if (v) VecDestroy(&v); }

  DenseVec operator - () const {
    Vec minus; VecDuplicate(v, &minus); VecCopy(v, minus);
    VecScale(minus, -1.0);
    return minus;
  }

  DenseVec& operator *= (double scale) {
    VecScale(v, scale);
    return *this;
  }

  auto size() const {
    int isize;
    VecGetSize(v, &isize);
    return isize;
  }

  double operator[](int i) const {
    double val;
    VecGetValues(v, 1, &i, &val);
    return val;
  }

  double operator[](size_t i) const {
    int i_int = static_cast<int>(i);
    double val;
    VecGetValues(v, 1, &i_int, &val);
    return val;
  }

  void setValue(int i, double val) {
    VecSetValues(v, 1, &i, &val, INSERT_VALUES);
  }

  void setValue(size_t i, double val) {
    int i_int = static_cast<int>(i);
    VecSetValues(v, 1, &i_int, &val, INSERT_VALUES);
  }

  void add(double val, const DenseVec& w) {
    VecAXPY(v, val, w.v);
  }

  std::vector<double> getValues() const {
    size_t sz = static_cast<size_t>(size());
    std::vector<double> vout(sz);
    std::vector<int> allints(sz);
    for (size_t i=0; i < sz; ++i) {
      allints[i] = static_cast<int>(i);
    }
    int sz_int = static_cast<int>(sz);
    VecGetValues(v, sz_int, &allints[0], &vout[0]);
    return vout;
  }

  void print(std::string first="") const {
    if (first.size()) {
      std::cout << first << ": ";
    }
    VecView(v, PETSC_VIEWER_STDOUT_SELF);
  }

  Vec v;
};

DenseVec DenseMat::operator*(const DenseVec& v) const {
  Vec out; VecDuplicate(v.v, &out);
  MatMult(A, v.v, out);
  return out;
}

DenseVec DenseMat::solve(const DenseVec& v) const {
  Vec out; VecDuplicate(v.v, &out);
  MatLUFactor(A, NULL, NULL, NULL); // not efficient if done a lot
  MatSolve(A, v.v, out);
  return out;
}

DenseMat DenseMat::PtAP(const DenseMat& P) const {
  Mat pAp;
  MatPtAP(A, P.A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &pAp);
  return pAp;
}

double dot(const DenseVec& a, const DenseVec& b) {
  double d;
  VecDot(a.v, b.v, &d);
  return d;
}

DenseVec operator+ (const DenseVec& a, double b) {
  Vec c; VecDuplicate(a.v, &c);
  VecSet(c, b);
  VecAXPY(c, 1.0, a.v);
  return c;
}

DenseVec operator* (const DenseVec& a, const DenseVec& b) {
  Vec c; VecDuplicate(a.v, &c);
  VecPointwiseMult(c, a.v, b.v);
  return c;
}

DenseVec operator/ (const DenseVec& a, const DenseVec& b) {
  Vec c; VecDuplicate(a.v, &c);
  VecPointwiseDivide(c, a.v, b.v);
  return c;
}

DenseVec abs(const DenseVec& a) {
  Vec absa; VecDuplicate(a.v, &absa); VecCopy(a.v, absa);
  VecAbs(absa);
  return absa;
}

double sum(const DenseVec& a) {
  double s;
  VecSum(a.v, &s);
  return s;
}

double norm(const DenseVec& a) {
  double n;
  VecNorm(a.v, NORM_2, &n);
  return n;
}

auto eigh(const DenseMat& Adense)
{
  auto [isize, jsize] = Adense.size();
  SLIC_ERROR_IF(isize!=jsize, "Eig must be called for symmetric matrices");

  const Mat& A = Adense.A;

  EPS eps;
  EPSCreate(PETSC_COMM_SELF, &eps);
  EPSSetOperators(eps,A,NULL);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetDimensions(eps, isize, PETSC_DETERMINE, PETSC_DETERMINE);
  EPSSetFromOptions(eps);

  EPSSolve(eps);

  EPSType type;
  EPSGetType(eps, &type);
  PetscPrintf(PETSC_COMM_SELF," Solution method: %s\n\n",type);
  EPSGetDimensions(eps, &jsize, NULL, NULL);
  SLIC_WARNING_IF(isize!=jsize, "The requested and achieved number of eigenvalues do not match it eigh call");
  PetscPrintf(PETSC_COMM_SELF," Number of requested eigenvalues: %" PetscInt_FMT "\n",isize);
  PetscPrintf(PETSC_COMM_SELF," Number of requested eigenvalues: %" PetscInt_FMT "\n",jsize);

  DenseVec eigenvalues(isize);
  std::vector<DenseVec> eigenvectors;
  for (int i=0; i < isize; ++i) {
    eigenvectors.emplace_back(isize);
    double eigenvalue;
    EPSGetEigenpair(eps, i, &eigenvalue, PETSC_NULLPTR, eigenvectors[static_cast<size_t>(i)].v, PETSC_NULLPTR);
    eigenvalues.setValue(i, eigenvalue);
  }

  EPSDestroy(&eps);
  return std::make_pair(std::move(eigenvalues), std::move(eigenvectors));
}
