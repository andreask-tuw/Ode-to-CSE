#include <iostream>
#include <fstream> 
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

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
private:
  double R;
  double C;

public:
  RC(double resistency, double capacity) : R(resistency), C(capacity) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    double u_c = x(0);
    double t = x(1);
    double u_0 = std::cos(100 * M_PI * t);

    f(0) = (u_0 - u_c) / (R * C);
    f(1) = 1.0;
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    double t = x(1);
    
    df = 0.0;
    df(0,0) = -1.0 / (R * C);
    df(0,1) = -100 * M_PI * std::sin(100 * M_PI * t) / (R * C);
  }
};


int main()
{
  double tend = 4*M_PI;
  int steps = 100;
  double tau = tend/steps;

  // Vector<> y = { 1, 0 };  // initializer list
  // auto rhs = std::make_shared<MassSpring>(1.0, 1.0);
  
  Vector<> y = { 0, 0 };  // initializer list
  auto rhs = std::make_shared<RC>(1.0, 1.0);

  // ExplicitEuler stepper(rhs);
  ImprovedEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  // CrankNicholson stepper(rhs);

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
