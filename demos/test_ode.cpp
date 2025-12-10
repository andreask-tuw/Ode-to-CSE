#include <iostream>
#include <fstream> 
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <autodiff.hpp>
#include <implicitRK.hpp>

using namespace ASC_ode;

class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};


class RC : public NonlinearFunction
{
    double resistency;
    double capacity;
    double omega;

public:
    RC(double R, double C, double w) 
        : resistency(R), capacity(C), omega(w) {}

    // Dimension is 2: [0] is Voltage, [1] is Time
    size_t dimX() const override { return 2; }
    size_t dimF() const override { return 2; }

    // y' = f(y)
    void evaluate(VectorView<double> x, VectorView<double> f) const override
    {
        double U0 = std::cos(omega * x(1));
        f(0) = (U0 - x(0)) / (resistency * capacity);   
        // dt/dt = 1
        f(1) = 1.0;
    }

    void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
    {
        double t = x(1);
        df = 0.0;
        df(0, 0) = -1.0 / (resistency * capacity);
        df(0, 1) = -(omega / (resistency * capacity)) * std::sin(omega * t);
        
        // df(1)/... are all 0 because dt/dt = 1 is constant
    }
};

class PendulumAD : public NonlinearFunction
{
private:
  double m_length;
  double m_gravity;

public:
  PendulumAD(double length, double gravity=9.81) : m_length(length), m_gravity(gravity) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    T_evaluate<double>(x, f);
  }

  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    Vector<AutoDiff<2>> x_ad(2);
    Vector<AutoDiff<2>> f_ad(2);

    x_ad(0) = Variable<0>(x(0));
    x_ad(1) = Variable<1>(x(1));
    T_evaluate<AutoDiff<2>>(x_ad, f_ad);

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
         df(i,j) = f_ad(i).deriv()[j];
  }

  template <typename T>
  void T_evaluate (VectorView<T> x, VectorView<T> f) const

  {
    f(0) = x(1);
    f(1) = T(-m_gravity/m_length)*sin(x(0));
  }
};


int main()
{
  double tend = 2*M_PI;
  int steps = 10000;
  double tau = tend/steps;

  // Vector<> y = { 1, 0 };  // initializer list
  // auto rhs = std::make_shared<MassSpring>(1.0, 1.0);
  // double omega = 50*M_PI;
  // double time_const = 1/omega;
  // double R = 30, C = time_const/R;


  // Vector<> y = { 0, 0 };  // initializer list
  // auto rhs = std::make_shared<RC>(R, C, omega);

  Vector<> y = { 2, 0 };  // initializer list
  auto rhs = std::make_shared<PendulumAD>(1);

  // ExplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  CrankNicholson stepper(rhs);

  // RungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);

  // Gauss3c .. points tabulated, compute a,b:
  auto [Gauss3a,Gauss3b] = computeABfromC (Gauss3c);
  ImplicitRungeKutta stepper(rhs, Gauss3a, Gauss3b, Gauss3c);


  /*
  // arbitrary order Gauss-Legendre
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussLegendre(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */

  /* 
  // arbitrary order Radau
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussRadau(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */


  std::ofstream outfile ("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.doStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}
