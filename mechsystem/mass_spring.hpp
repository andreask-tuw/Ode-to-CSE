#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <array>


using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};


// adding distance constraint between two connectors - here only storing the data
class DistanceConstraint
{
public:
  std::array<Connector,2> connectors;
  double distance;
  DistanceConstraint (Connector c1, Connector c2, double dist)
    : connectors{{c1,c2}}, distance(dist) { }
};


template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<DistanceConstraint> m_joints;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s); 
    return m_springs.size()-1;
  }

  size_t addJoint (DistanceConstraint j)
  {
    m_joints.push_back (j);
    return m_joints.size()-1;
  }

  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }
  auto & joints() { return m_joints; }

  void getState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(i) = m_masses[i].pos;
        dvalmat.row(i) = m_masses[i].vel;
        ddvalmat.row(i) = m_masses[i].acc;
      }
  }

  void setState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(i);
        m_masses[i].vel = dvalmat.row(i);
        m_masses[i].acc = ddvalmat.row(i);
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}


template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_Function (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return D*mss.masses().size() + mss.joints().size(); }
  virtual size_t dimF() const override { return D*mss.masses().size() + mss.joints().size(); }
  
  virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f = 0.0;
    //number of masses and joints
    size_t nm = mss.masses().size();
    size_t nj = mss.joints().size();

  // x - solution vector is now cosisting of positions; lambdas
  auto xmat    = x.range(0, nm*D).asMatrix(nm, D);
  auto lambdas = x.range(nm*D, nm*D + nj);

  // f - force vector now consisting of forces; constraint residuals
  auto fmat = f.range(0, nm*D).asMatrix(nm, D);
  auto gres = f.range(nm*D, nm*D + nj);

    // auto xmat = x.asMatrix(mss.masses().size(), D);
    // auto fmat = f.asMatrix(mss.masses().size(), D);

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass*mss.getGravity();

    for (auto spring : mss.springs())
      {
        auto [c1,c2] = spring.connectors;
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        double force = spring.stiffness * (norm(p1-p2)-spring.length);
        Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1);
        if (c1.type == Connector::MASS)
          fmat.row(c1.nr) += force*dir12;
        if (c2.type == Connector::MASS)
          fmat.row(c2.nr) -= force*dir12;
      }

    for (size_t j = 0; j < nj; j++)
    // for (auto joint : mss.joints())
    {
      auto joint = mss.joints()[j];
      auto [c1,c2] = joint.connectors;
      Vec<D> p1, p2; // start and end point of the spring
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos; //fix coordinates of the fix
      else
        p1 = xmat.row(c1.nr); // coord of a mass
      if (c2.type == Connector::FIX) //ending point
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);

      // double force = spring.stiffness * (norm(p1-p2)-spring.length);
      // Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1);
      // if (c1.type == Connector::MASS)
      //   fmat.row(c1.nr) += force*dir12;
      // if (c2.type == Connector::MASS)
      //   fmat.row(c2.nr) -= force*dir12;


      Vec<D> diff = p1 - p2;
      double lambda = lambdas(j);
      if (c1.type == Connector::MASS)
          fmat.row(c1.nr) += (2 * lambda) * diff;
      if (c2.type == Connector::MASS)
          fmat.row(c2.nr) -= (2 * lambda) * diff;
      
      gres(j) = dot(diff, diff) - joint.distance * joint.distance;
      
    }


    // from forces to accelerations
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0/mss.masses()[i].mass;



  }
  
  virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    // TODO: exact differentiation
    double eps = 1e-8;
    Vector<> xl(dimX()), xr(dimX()), fl(dimF()), fr(dimF());
    for (size_t i = 0; i < dimX(); i++)
      {
        xl = x;
        xl(i) -= eps;
        xr = x;
        xr(i) += eps;
        evaluate (xl, fl);
        evaluate (xr, fr);
        df.col(i) = 1/(2*eps) * (fr-fl);
      }
  }
  
};

#endif
