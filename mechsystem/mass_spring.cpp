#include "mass_spring.hpp"
#include "Newmark.hpp"

int main()
{
  MassSpringSystem<2> mss;
  mss.setGravity( {0,-9.81} );
  auto fA = mss.addFix( { { 0.0, 0.0 } } );
  auto mA = mss.addMass( { 1, { 1.0, 0.0 } } );
  mss.addSpring ( { 1, 10, { fA, mA } }  );

  auto mB = mss.addMass( { 1, { 2.0, 0.0 } } );
  mss.addSpring ( { 1, 20, { mA, mB } } );

  mss.addJoint( DistanceConstraint(mA, mB, 1.0) );


  std::cout << "mss: " << std::endl << mss << std::endl;


  double tend = 10;
  double steps = 1000;

  // adjust size of the system to include also lambdas for distance constraints
  size_t nm = mss.masses().size();
  size_t nj = mss.joints().size();

  Vector<> x(2*nm + nj);
  Vector<> dx(2*nm + nj);
  Vector<> ddx(2*nm + nj);


  // Vector<> x(2*mss.masses().size());
  // Vector<> dx(2*mss.masses().size());
  // Vector<> ddx(2*mss.masses().size());

  auto mss_func = std::make_shared<MSS_Function<2>> (mss);
  auto mass = std::make_shared<IdentityFunction> (x.size());

  // mss.getState (x, dx, ddx);

  mss.getState( x.range(0, 2*nm),
              dx.range(0, 2*nm),
              ddx.range(0, 2*nm) );

  // initialize lambda part
  x.range(2*nm, 2*nm+nj) = 0.0;
  dx.range(2*nm, 2*nm+nj) = 0.0;
  ddx.range(2*nm, 2*nm+nj) = 0.0;

  
  // SolveODE_Newmark(tend, steps, x, dx,  mss_func, mass,
  //                  [](double t, VectorView<double> x) { std::cout << "t = " << t
  //                                                            << ", x = " << Vec<4>(x) << std::endl; });

  SolveODE_Newmark(tend, steps, x, dx, mss_func, mass,
  [nm,nj](double t, VectorView<double> x)
  {
    std::cout << "t = " << t
              << ", pos = " << Vec<4>(x.range(0,2*nm))
              << ", lambdas = " << x.range(2*nm, 2*nm+nj)
              << std::endl;
  });

}
