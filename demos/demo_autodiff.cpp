#include <iostream>
#include <autodiff.hpp>


using namespace ASC_ode;


template <typename T>
T func1 (T x, T y)
{
  return x * sin(y);
  // return 1e6 + y;
}

template <typename T>
T f (T x, T y)
{
  return x * y;
}

template <typename T>
T func2 (T x, T y)
{
  return x * exp(y);
}

template <typename T>
T func3 (T x, T y)
{
  return log(y);
}


int main()
{
  double x = 1, y = 2;
  AutoDiff<2> adx = Variable<0>(x);
  AutoDiff<2> ady = Variable<1>(y);

  std::cout << "adx = " << adx << std::endl;
  std::cout << "ady = " << ady << std::endl;

  AutoDiff<2> prod = adx * ady;
  std::cout << "prod = " << prod << std::endl;

  std::cout << "func1(adx, ady) = " << func1(adx, ady) << std::endl;
  std::cout << "func2(adx, ady) = " << func2(adx, ady) << std::endl;
  std::cout << "func3(adx, ady) = " << func3(adx, ady) << std::endl;
  // printf("\n");
  // std::cout << "f(adx, ady) = " << f(adx, ady) << std::endl;

  double eps = 1e-8;
  printf("-------------------f1-------------------\n");
  std::cout << "numdiff df/dx= " << (func1(x + eps, y) - func1(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func1(x, y + eps) - func1(x, y-eps)) / (2*eps) << std::endl;
  printf("-------------------f2-------------------\n");
  std::cout << "numdiff df/dx= " << (func2(x + eps, y) - func2(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func2(x, y + eps) - func2(x, y-eps)) / (2*eps) << std::endl;
  printf("-------------------f3-------------------\n");
  std::cout << "numdiff df/dx = " << (func3(x + eps, y) - func3(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func3(x, y + eps) - func3(x, y-eps)) / (2*eps) << std::endl;
  printf("----------------------------------------\n");
  {
    // we can do second derivatives:
    printf("2nd derivatives:\n");
    AutoDiff<1, AutoDiff<1>> addx{Variable<0>(2)};
    std::cout << "addx = " << addx << std::endl;
    // func = x*x
    // func' = 2*x
    // func'' = 2
    std::cout << "addx*addx = " << addx * addx << std::endl;

    std::cout << "sin(addx) = " << sin(addx) << std::endl;
  }
  return 0;
}