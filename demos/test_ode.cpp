#include <iostream>
#include <fstream> 
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <autodiff.hpp>
#include <implicitRK.hpp>
#include <explicitRK.hpp>

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

  Vector<> y = { 1, 0 };  // initializer list
  Vector<> y0 = y;       // save initial condition
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);


  /*** EXPLICIT EULER ***/
  ExplicitEuler stepper1(rhs);

  std::string filename1 = "output_test_ode_ExplicitEuler.txt";
  std::ofstream outfile1(filename1);

  std::cout << "# method: " << "Explicit Euler" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile1 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper1.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile1 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }

  // reset initial condition for each method
  y = y0;


  /*** IMPROVED EULER ***/
  ImprovedEuler stepper2(rhs);

  std::string filename2 = "output_test_ode_ImprovedEuler.txt";
  std::ofstream outfile2(filename2);

  std::cout << "# method: " << "Improved Euler" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile2 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper2.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile2 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }


  // reset initial condition for each method
  y = y0;


  /*** IMPLICIT EULER ***/

  ImplicitEuler stepper3(rhs);

  std::string filename3 = "output_test_ode_ImplicitEuler.txt";
  std::ofstream outfile3(filename3);

  std::cout << "# method: " << "Implicit Euler" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile3 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper3.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile3 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }

  // reset initial condition for each method
  y = y0;


  /*** CRANK NICHOLSON ***/

  CrankNicholson stepper4(rhs);

  std::string filename4 = "output_test_ode_CrankNicholson.txt";
  std::ofstream outfile4(filename4);

  std::cout << "# method: " << "Crank Nicholson" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile4 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper4.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile4 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
  

  // reset initial condition for each method
  y = y0;


  /*** IMPLICIT RUNGE-KUTTA ***/

  double omega = 50*M_PI;
  double time_const = 1/omega;
  double R = 30, C = time_const/R;

  // Gauss3c .. points tabulated, compute a,b:
  auto [Gauss3a,Gauss3b] = computeABfromC (Gauss3c);
  ImplicitRungeKutta stepper5(rhs, Gauss3a, Gauss3b, Gauss3c);

  std::string filename5 = "output_test_ode_IRK.txt";
  std::ofstream outfile5(filename5);

  std::cout << "# method: " << "Implicit Runge Kutta" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile5 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper5.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile5 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }


  // reset initial condition for each method
  y = y0;


  /*** EXPLICIT RUNGE-KUTTA ***/

  // Use a valid explicit tableau (classical RK4).
  auto [ERK4a, ERK4b, ERK4c] = ERK_RK4_Tableau();
  ExplicitRungeKutta stepper6(rhs, ERK4a, ERK4b, ERK4c);

  std::string filename6 = "output_test_ode_ERK.txt";
  std::ofstream outfile6(filename6);

  std::cout << "# method: " << "Explicit Runge Kutta" << "\n";
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile6 << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper6.doStep(tau, y);
    // std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile6 << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }


  // Vector<> y_RC = { 0, 0 };  // initializer list
  // auto rhs_RC = std::make_shared<RC>(R, C, omega);

  // Vector<> y_AD = { 2, 0 };  // initializer list
  // auto rhs_AD = std::make_shared<PendulumAD>(1);


  // RungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);
  // ImplicitRungeKutta stepper_IRK(rhs, Gauss2a, Gauss2b, Gauss2c);




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


  
}
